import cv2
import json
import os

IMAGES = [
    ('like',  'like.jpg'),
    ('rock',  'rock.jpg'),
    ('peace', 'peace.jpg'),
]

OUTPUT = 'annot-FArneH.json'
DISPLAY_H = 800

annotations = {}
drawing = False
start_x = start_y = end_x = end_y = 0


def mouse_callback(event, x, y, flags, param):
    global drawing, start_x, start_y, end_x, end_y
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y
        end_x, end_y = x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        end_x, end_y = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_x, end_y = x, y


for label, filename in IMAGES:
    if not os.path.exists(filename):
        print(f'Datei nicht gefunden: {filename}')
        continue

    img = cv2.imread(filename)
    orig_h, orig_w = img.shape[:2]

    scale = DISPLAY_H / orig_h
    display_w = int(orig_w * scale)
    img_display = cv2.resize(img, (display_w, DISPLAY_H))

    cv2.namedWindow(label)
    cv2.setMouseCallback(label, mouse_callback)
    start_x = start_y = end_x = end_y = 0

    print(f'\n[{label}] Bounding Box um die Hand ziehen, dann ENTER druecken.')

    while True:
        frame = img_display.copy()

        if start_x != end_x and start_y != end_y:
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

        cv2.putText(frame, f'Geste: {label} | Box ziehen, dann ENTER',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow(label, frame)

        key = cv2.waitKey(20) & 0xFF
        if key == 13 or key == ord('\r'):
            break
        if key == ord('q'):
            cv2.destroyAllWindows()
            exit()

    cv2.destroyAllWindows()

    # Koordinaten normalisieren
    x1 = min(start_x, end_x) / scale
    y1 = min(start_y, end_y) / scale
    x2 = max(start_x, end_x) / scale
    y2 = max(start_y, end_y) / scale

    bx = x1 / orig_w
    by = y1 / orig_h
    bw = (x2 - x1) / orig_w
    bh = (y2 - y1) / orig_h

    uid = filename.split('.')[0]
    annotations[uid] = {
        'bboxes': [[round(bx, 8), round(by, 8), round(bw, 8), round(bh, 8)]],
        'labels': [label],
        'landmarks': [],
        'leading_conf': 1.0,
        'leading_hand': 'right',
        'user_id': ''
    }
    print(f'  -> gespeichert: bbox=[{bx:.4f}, {by:.4f}, {bw:.4f}, {bh:.4f}]')

with open(OUTPUT, 'w') as f:
    json.dump(annotations, f, indent=4)

print(f'\nFertig! Annotationen gespeichert in {OUTPUT}')
