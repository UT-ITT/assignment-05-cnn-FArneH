import cv2
import json
import os
import subprocess
import sys
import time
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, RandomFlip, RandomContrast
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical
from pynput.keyboard import Key, Controller
from sklearn.model_selection import train_test_split

MODEL_PATH = 'gesture_model.keras'

# Datensatz suchen (neben dem Skript, eine Ebene höher oder in Downloads)
_script_dir = os.path.dirname(os.path.abspath(__file__))
_candidates = [
    os.path.join(_script_dir, 'gesture_dataset_sample'),
    os.path.join(_script_dir, '..', 'gesture_dataset_sample'),
    os.path.join(os.path.expanduser('~'), 'Downloads', 'gesture_dataset_sample'),
]
DATASET_PATH = next((p for p in _candidates if os.path.isdir(p)), _candidates[0])

CATEGORIES = ['like', 'dislike', 'stop', 'peace']

IMG_SIZE = 64
CONFIDENCE = 0.75  # Mindestkonfidenz für eine Aktion
COOLDOWN = 2.0     # Sekunden zwischen zwei Aktionen


def build_model(num_classes):
    model = Sequential([
        RandomFlip('horizontal'),
        RandomContrast(0.1),
        Conv2D(64, (9, 9), activation='leaky_relu', padding='same',
               input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        MaxPooling2D((4, 4), padding='same'),
        Conv2D(32, (5, 5), activation='leaky_relu', padding='same'),
        MaxPooling2D((3, 3), padding='same'),
        Conv2D(32, (3, 3), activation='leaky_relu', padding='same'),
        MaxPooling2D((2, 2), padding='same'),
        Dropout(0.2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax'),
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def crop_bbox(img, bbox):
    x1 = int(bbox[0] * img.shape[1])
    y1 = int(bbox[1] * img.shape[0])
    w = int(bbox[2] * img.shape[1])
    h = int(bbox[3] * img.shape[0])
    crop = img[y1:y1 + h, x1:x1 + w]
    return None if crop.size == 0 else cv2.resize(crop, (IMG_SIZE, IMG_SIZE))


def load_category(cat, idx, annotations):
    images, labels = [], []
    for filename in os.listdir(f'{DATASET_PATH}/{cat}'):
        uid = filename.split('.')[0]
        if uid not in annotations:
            continue
        img = cv2.imread(f'{DATASET_PATH}/{cat}/{filename}')
        if img is None:
            continue
        ann = annotations[uid]
        for i, bbox in enumerate(ann['bboxes']):
            if ann['labels'][i] != cat:
                continue
            crop = crop_bbox(img, bbox)
            if crop is not None:
                images.append(crop)
                labels.append(idx)
    return images, labels


def load_data():
    images, labels = [], []
    for idx, cat in enumerate(CATEGORIES):
        with open(f'{DATASET_PATH}/_annotations/{cat}.json') as f:
            annotations = json.load(f)
        imgs, lbls = load_category(cat, idx, annotations)
        images.extend(imgs)
        labels.extend(lbls)
    return np.array(images), np.array(labels)


def train_model():
    print('Daten laden...')
    X, y = load_data()
    X = X.astype('float32') / 255.0
    y = to_categorical(y, len(CATEGORIES))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print('Modell trainieren...')
    model = build_model(len(CATEGORIES))
    model.fit(
        X_train, y_train,
        batch_size=8,
        epochs=50,
        validation_data=(X_test, y_test),
        callbacks=[EarlyStopping(patience=3), ReduceLROnPlateau(patience=2, min_lr=1e-4)],
        verbose=1
    )

    _, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f'Genauigkeit: {acc:.2f}')
    model.save(MODEL_PATH)
    return model


def detect_hand(roi):
    # Hautfarbe im HSV-Raum erkennen
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 20, 60]), np.array([25, 255, 255]))
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 800:  # zu klein, wahrscheinlich Rauschen
        return None

    x, y, w, h = cv2.boundingRect(largest)
    pad = 12
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(roi.shape[1], x + w + pad)
    y2 = min(roi.shape[0], y + h + pad)
    crop = roi[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


def trigger_action(gesture, keyboard):
    # auf macOS funktionieren die pynput-Lautstärketasten nicht zuverlässig
    if gesture == 'like':
        if sys.platform == 'darwin':
            subprocess.run(['osascript', '-e',
                'set volume output volume (output volume of (get volume settings) + 10)'],
                capture_output=True)
        else:
            keyboard.press(Key.media_volume_up)
            keyboard.release(Key.media_volume_up)
    elif gesture == 'dislike':
        if sys.platform == 'darwin':
            subprocess.run(['osascript', '-e',
                'set volume output volume (output volume of (get volume settings) - 10)'],
                capture_output=True)
        else:
            keyboard.press(Key.media_volume_down)
            keyboard.release(Key.media_volume_down)
    elif gesture == 'stop':
        keyboard.press(Key.media_play_pause)
        keyboard.release(Key.media_play_pause)
    elif gesture == 'peace':
        keyboard.press(Key.media_next)
        keyboard.release(Key.media_next)


def get_box_style(gesture, confidence):
    if gesture is None:
        return (160, 160, 160), 'keine Hand'
    if confidence >= CONFIDENCE:
        return (0, 220, 0), f'{gesture} {confidence:.0%}'
    return (0, 140, 255), f'{gesture} {confidence:.0%}'


def run():
    # Modell laden oder neu trainieren
    if os.path.exists(MODEL_PATH):
        print('Gespeichertes Modell wird geladen...')
        model = load_model(MODEL_PATH)
    else:
        model = train_model()

    keyboard = Controller()
    cap = cv2.VideoCapture(0)
    last_action = 0

    print('\nSteuerung:')
    print('  like    -> Lauter')
    print('  dislike -> Leiser')
    print('  stop    -> Play / Pause')
    print('  peace   -> Nächster Titel')
    print('Q zum Beenden\n')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        fh, fw = frame.shape[:2]

        # ROI in der Bildmitte berechnen
        size = int(min(fw, fh) * 0.45)
        cx, cy = fw // 2, fh // 2
        rx1, ry1 = cx - size // 2, cy - size // 2
        rx2, ry2 = cx + size // 2, cy + size // 2

        roi = frame[ry1:ry2, rx1:rx2]
        hand_crop = detect_hand(roi)

        gesture = None
        confidence = 0.0
        probs = np.zeros(len(CATEGORIES))

        if hand_crop is not None:
            inp = cv2.resize(hand_crop, (IMG_SIZE, IMG_SIZE))
            inp = np.expand_dims(inp.astype('float32') / 255.0, 0)
            probs = model.predict(inp, verbose=0)[0]
            idx = int(np.argmax(probs))
            gesture = CATEGORIES[idx]
            confidence = float(probs[idx])

        # Aktion auslösen wenn Konfidenz hoch genug und Cooldown abgelaufen
        now = time.time()
        if gesture and confidence >= CONFIDENCE and now - last_action >= COOLDOWN:
            trigger_action(gesture, keyboard)
            last_action = now
            print(f'  {gesture} ({confidence:.0%})')

        # ROI-Box zeichnen
        box_color, label = get_box_style(gesture, confidence)
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), box_color, 2)
        cv2.putText(frame, 'Hand hierher halten', (rx1, ry1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
        cv2.putText(frame, label, (12, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, box_color, 2)

        # Konfidenzbalken unten links
        for i, (cat, prob) in enumerate(zip(CATEGORIES, probs)):
            by = fh - 30 - i * 22
            cv2.rectangle(frame, (10, by - 14), (10 + int(prob * 120), by), (80, 180, 80), -1)
            cv2.putText(frame, f'{cat} {prob:.0%}', (135, by),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

        # Cooldown-Balken ganz unten
        elapsed = min(now - last_action, COOLDOWN)
        cv2.rectangle(frame, (0, fh - 6), (int(elapsed / COOLDOWN * fw), fh), (0, 200, 0), -1)

        cv2.imshow('Media Controller - Q zum Beenden', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run()
