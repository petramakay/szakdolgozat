import cv2
import numpy as np
import rtde_receive
import rtde_control
from typing import List, Tuple
import time

# Robot inicializálása
robot_ip = "10.150.0.1"
rtde_c = rtde_control.RTDEControlInterface(robot_ip)
rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)

# Robot home pozícióba mozgatása
rtde_c.moveL([-0.10538271612203795, -0.45699951056036625, 0.30453289071494344,
              0.15828186296338534, -2.2117516285928693, 2.1983071443992195], 0.06, 0.06)

current_pose = rtde_r.getActualTCPPose()
print("Current TCP Pose:", current_pose)

robot_coordinates = []


def transform_coordinates(pixel_points: List[Tuple[int, int]]) -> List[Tuple[float, float]]:
    """
    Átalakítja a pixel koordinátákat robot koordinátákká affin transzformáció használatával.
    """
    ref_pixels = np.array([
        [628, 307, 1],
        [718, 335, 1],
        [711, 426, 1],
        [621, 397, 1],
        [721, 309, 1],
        [799, 314, 1],
        [787, 437, 1]
    ])

    ref_robot = np.array([
        [-0.17096802216745088, -0.2822605130761222],
        [-0.11107599806005546, -0.29851333904563787],
        [-0.11255847019751782, -0.35755271462652743],
        [-0.17275235823745647, -0.34153859191249136],
        [-0.11244621412764004, -0.2760906924091822],
        [-0.06313047631880779, -0.27571414328246074],
        [-0.06391195432021181, -0.3532935132486609]
    ])

    # Transzformációs mátrix számítása
    A = np.linalg.lstsq(ref_pixels, ref_robot, rcond=None)[0]

    # Input pontok átalakítása homogén koordinátákká
    points_array = np.array(pixel_points)
    pixel_points_homog = np.hstack([points_array, np.ones((len(pixel_points), 1))])

    # Transzformáció alkalmazása
    robot_points = pixel_points_homog @ A

    return list(map(tuple, robot_points))


def calculate_points(corners: List[tuple]) -> List[tuple]:
    """
    Kiszámolja egy négyszög oldalainak harmadolópontjait.
    """
    result = []
    n = len(corners)

    for i in range(n):
        result.append(corners[i])
        next_idx = (i + 1) % n

        first_third_x = int(corners[i][0] + (corners[next_idx][0] - corners[i][0]) / 3)
        first_third_y = int(corners[i][1] + (corners[next_idx][1] - corners[i][1]) / 3)
        result.append((first_third_x, first_third_y))

        second_third_x = int(corners[i][0] + 2 * (corners[next_idx][0] - corners[i][0]) / 3)
        second_third_y = int(corners[i][1] + 2 * (corners[next_idx][1] - corners[i][1]) / 3)
        result.append((second_third_x, second_third_y))

    return result


def filter_rectangles(contour, min_area=1000, max_area=20000, aspect_ratio_range=(0.3, 3.0)):
    """
    Szűrjük a négyszögeket terület és oldalarány alapján
    """
    area = cv2.contourArea(contour)
    if area < min_area or area > max_area:
        return False

    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h if h != 0 else 0
    if aspect_ratio < aspect_ratio_range[0] or aspect_ratio > aspect_ratio_range[1]:
        return False

    # Ellenőrizzük, hogy a négyszög nem túl közel van-e a kép széléhez
    image_margin = 50
    frame_height, frame_width = 720, 1280

    if x < image_margin or y < image_margin or \
            x + w > frame_width - image_margin or \
            y + h > frame_height - image_margin:
        return False

    return True


def is_quadrilateral(approx):
    """
    Ellenőrzi, hogy a kontúr négyszög-e
    """
    if len(approx) != 4:
        return False

    # Minimális oldalhossz ellenőrzése
    min_side_length = 20
    for i in range(4):
        p1 = approx[i][0]
        p2 = approx[(i + 1) % 4][0]
        side_length = np.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))
        if side_length < min_side_length:
            return False

    return True


def remove_duplicate_contours(contours, threshold_distance=20):
    """
    Eltávolítja a duplikált kontúrokat a középpontjuk távolsága alapján
    """
    if not contours:
        return []

    # Középpontok kiszámítása
    centers = []
    filtered_contours = []

    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Ellenőrizzük, hogy van-e már közeli középpont
            is_duplicate = False
            for existing_center in centers:
                distance = np.sqrt((cx - existing_center[0]) ** 2 + (cy - existing_center[1]) ** 2)
                if distance < threshold_distance:
                    is_duplicate = True
                    break

            if not is_duplicate:
                centers.append((cx, cy))
                filtered_contours.append(contour)

    return filtered_contours


def process_single_frame(frame):
    """
    Egyetlen képkocka feldolgozása
    """
    output = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )

    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    shapes = []
    all_points = []

    for contour in contours:
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if filter_rectangles(contour) and is_quadrilateral(approx):
            shapes.append(approx)

            corner_points = [point[0] for point in approx]
            corner_points.sort(key=lambda p: (p[1], p[0]))

            top = corner_points[:2]
            bottom = corner_points[2:]

            top.sort(key=lambda p: p[0])
            bottom.sort(key=lambda p: p[0], reverse=True)
            ordered_corners = top + bottom

            extended_points = calculate_points(ordered_corners)
            all_points.extend(extended_points)

            transformed_points = transform_coordinates(extended_points)
            robot_coordinates.append(transformed_points)

            cv2.drawContours(output, [approx], -1, (0, 255, 0), 2)

            for i, point in enumerate(extended_points):
                x, y = point
                robot_x, robot_y = transformed_points[i]

                if i % 3 == 0:
                    cv2.circle(output, (x, y), 5, (0, 0, 255), -1)
                    label = chr(65 + (i // 3))
                    cv2.putText(output, label, (x - 10, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                else:
                    cv2.circle(output, (x, y), 3, (255, 0, 0), -1)
                    edge_idx = i // 3
                    point_idx = i % 3
                    label = f"T{edge_idx + 1}_{point_idx}"
                    cv2.putText(output, label, (x - 10, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                print(f"Pont {label}:")
                print(f"  Pixel: ({x}, {y})")
                print(f"  Robot: ({robot_x:.6f}, {robot_y:.6f})")

    shapes = remove_duplicate_contours(shapes)

    return shapes, output, robot_coordinates


def move_robot(coordinates):
    print("Rajzolás megkezdése...")
    if rtde_r.isConnected() == False:
        rtde_r.reconnect()
    # Kinyúl
    rtde_c.moveL([-0.10536425699982244, -0.5278222252789551, 0.3045235131427382,
                  0.15832671973234444, -2.211808999638041, 2.1983103757415368], 0.06, 0.06)
    time.sleep(2.0)
    # Lenti pozíció
    if rtde_r.isConnected() == False:
        rtde_r.reconnect()
    rtde_c.moveL([-0.16313576407302574, -0.26710735843215216, -0.11372163201217396,
                  -0.18091484564018534, 3.1356715996878095, -0.05274949675824685], 0.06, 0.06)

    magassag = -0.11812805266124452
    rx = -0.18082320433672286
    ry = 3.1356571209171906
    rz = -0.052723214379207975
    rajzolomagassag = -0.1277202558173126

    # Minden alak összes pontjának összegyűjtése egy listába
    all_points = []
    for shape in robot_coordinates:
        all_points.extend(shape)

    # Minden pontból minden pontba mozgás
    for i, start_point in enumerate(all_points):
        for j, end_point in enumerate(all_points):
            if i < j:
                # Start ponthoz mozgás
                if rtde_r.isConnected() == False:
                    rtde_r.reconnect()
                rtde_c.moveL([start_point[0], start_point[1], magassag, rx, ry, rz], 0.07, 0.07)
                time.sleep(1.0)

                if rtde_r.isConnected() == False:
                    rtde_r.reconnect()
                rtde_c.moveL([start_point[0], start_point[1], rajzolomagassag, rx, ry, rz], 0.07, 0.07)

                # End ponthoz mozgás
                if rtde_r.isConnected() == False:
                    rtde_r.reconnect()
                rtde_c.moveL([end_point[0], end_point[1], rajzolomagassag, rx, ry, rz], 0.07, 0.07)
                time.sleep(1.0)

    time.sleep(1.0)
    # Lenti pozíció
    if rtde_r.isConnected() == False:
        rtde_r.reconnect()
    rtde_c.moveL([-0.16313576407302574, -0.26710735843215216, -0.11372163201217396,
                  -0.18091484564018534, 3.1356715996878095, -0.05274949675824685], 0.06, 0.06)
    time.sleep(2.0)
    # Kinyúl
    if rtde_r.isConnected() == False:
        rtde_r.reconnect()
    rtde_c.moveL([-0.10536425699982244, -0.5278222252789551, 0.3045235131427382,
                  0.15832671973234444, -2.211808999638041, 2.1983103757415368], 0.06, 0.06)
    time.sleep(2.0)
    # Kamera kép
    if rtde_r.isConnect() == False:
        rtde_r.reconnect()
    rtde_c.moveL([-0.10538271612203795, -0.45699951056036625, 0.30453289071494344,
                  0.15828186296338534, -2.2117516285928693, 2.1983071443992195], 0.06, 0.06)


def main():
    gst_str = (
        'udpsrc port=5000 caps = "application/x-rtp, media=(string)video, '
        'clock-rate=(int)90000, encoding-name=(string)H264, payload=(int)96" ! '
        'rtph264depay ! decodebin ! videoconvert ! appsink'
    )

    cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("Hiba: Nem sikerült kapcsolódni a stream-hez!")
        return

    print("Várakozás a képre...")

    try:
        ret, frame = cap.read()
        if ret:
            shapes, processed_frame, robot_coords = process_single_frame(frame)
            cv2.imshow("Detected Rectangles", processed_frame)
            cv2.waitKey(0)

            valasz = input("Szeretné mozgatni a robotot? (i/n): ").lower()

            if valasz == 'i':
                move_robot(robot_coords)
            elif valasz == 'n':
                print("Robot mozgatása kihagyva.")
            else:
                print("Érvénytelen válasz. A robot nem fog mozogni.")

        else:
            print("Nem sikerült képet olvasni a stream-ből.")

    except Exception as e:
        print(f"Hiba történt: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()