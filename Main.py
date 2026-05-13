import cv2
import numpy as np
import mediapipe as mp
import pytesseract
import re
import time


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
frame = cv2.flip(frame, 1)
h, w, _ = frame.shape
canvas = np.zeros((h, w), dtype=np.uint8)
prev_x, prev_y = None, None

expression = ""
result = ""
result_shown = False
selected_operator = None
last_calc_time = 0

#  OPERATOR BOX POSITIONS
operator_boxes = {
    "+": (50, 100, 150, 200),
    "-": (200, 100, 300, 200),
    "*": (350, 100, 450, 200),
    "/": (500, 100, 600, 200)
}


#  FINGER DETECTION
def get_fingers_up(lms):
    tips = [4, 8, 12, 16, 20]
    fingers = []

    fingers.append(1 if lms.landmark[tips[0]].x < lms.landmark[tips[0] - 2].x else 0)
    for i in range(1, 5):
        fingers.append(1 if lms.landmark[tips[i]].y < lms.landmark[tips[i] - 2].y else 0)

    return fingers


#  CLEAN OCR
def clean_text(t):
    t = t.strip()
    t = re.sub(r'[^0-9]', '', t)
    return t


# SOLVE
def solve(n1, n2, op):
    try:
        if op == "+": return str(n1 + n2)
        if op == "-": return str(n1 - n2)
        if op == "*": return str(n1 * n2)
        if op == "/": return str(round(n1 / n2, 2)) if n2 != 0 else "DIV 0"
    except:
        return "Err"


#  OCR + CALC
def extract_and_calculate():
    global expression, result, result_shown, last_calc_time

    if time.time() - last_calc_time < 2:
        return False
    last_calc_time = time.time()

    pts = cv2.findNonZero(canvas)
    if pts is None:
        return False

    x, y, bw, bh = cv2.boundingRect(pts)
    if bw < 50 or bh < 50:
        return False

    crop = canvas[y:y+bh, x:x+bw]
    blur = cv2.GaussianBlur(crop, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv = cv2.bitwise_not(th)
    big = cv2.resize(inv, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

    raw = pytesseract.image_to_string(big, config='--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789')
    cleaned = clean_text(raw)
    print("RAW:", raw, " CLEAN:", cleaned)

    if cleaned.isnumeric():
        expression = cleaned
        if len(cleaned) >= 2:
            n1 = int(cleaned[:-1])
            n2 = int(cleaned[-1])
            result = solve(n1, n2, selected_operator)
            result_shown = True
            return True
    return False


#  MAIN LOOP
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    frame_overlay = frame.copy()

    # Draw Operator Boxes
    for op, (x1, y1, x2, y2) in operator_boxes.items():
        color = (0, 255, 0) if selected_operator == op else (255, 255, 255)
        cv2.rectangle(frame_overlay, (x1, y1), (x2, y2), color, 3)
        cv2.putText(frame_overlay, op, (x1 + 30, y1 + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)

    if results.multi_hand_landmarks:
        lms = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame_overlay, lms, mp_hands.HAND_CONNECTIONS)

        fingers = get_fingers_up(lms)
        t, i, m, r, p = fingers
        total = sum(fingers)

        cx = int(lms.landmark[8].x * w)
        cy = int(lms.landmark[8].y * h)

        #OPERATOR SELECTION
        if total == 1 and i:
            for op, (x1, y1, x2, y2) in operator_boxes.items():
                if x1 < cx < x2 and y1 < cy < y2:
                    selected_operator = op
                    canvas[:] = 0
                    expression = ""
                    result = ""
                    result_shown = False

        # DRAW TWO NUMBERS
        if selected_operator is not None and total == 1 and i:
            cv2.circle(frame_overlay, (cx, cy), 12, (0, 255, 0), -1)
            if prev_x is not None:
                cv2.line(canvas, (prev_x, prev_y), (cx, cy), 255, 25)
            prev_x, prev_y = cx, cy

        #  CLEAR
        elif total == 1 and t:
            canvas[:] = 0
            prev_x = prev_y = None

        #  CALCULATE
        elif total == 3 and i and m and r:
            extract_and_calculate()

        else:
            prev_x = prev_y = None

    #  SHOW RESULT
    if result_shown:
        cv2.rectangle(frame_overlay, (0, h - 120), (w, h), (0, 0, 0), -1)
        cv2.putText(frame_overlay, f"{expression[0]} {selected_operator} {expression[1]} = {result}",
                    (30, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (0, 255, 255), 5)

    color_canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    final = cv2.addWeighted(frame_overlay, 0.7, color_canvas, 0.3, 0)

    cv2.putText(final, "Hover finger on + - * / to select operator",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

    cv2.imshow("Air Calculator with Operator Selection", final)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
