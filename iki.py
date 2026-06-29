import cv2
import mediapipe as mp
import pyautogui
import math
import numpy as np
import time

# Konfigurasi
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()
pyautogui.PAUSE = 0 

cap = cv2.VideoCapture(0)
w_cam, h_cam = 640, 480
cap.set(3, w_cam)
cap.set(4, h_cam)

PALM_IDS = [0, 5, 9, 13, 17]
DEADZONE = 20
PINCH_START = 24
def get_fingers_status(hand_lms):
    # status[0]=Telunjuk, status[1]=Tengah, status[2]=Manis, status[3]=Kelingking
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    status = []
    for tip, pip in zip(tips, pips):
        if hand_lms.landmark[tip].y < hand_lms.landmark[pip].y:
            status.append(True)
        else:
            status.append(False)
    return status 

def is_gripping(hand_lms):
    f_status = get_fingers_status(hand_lms)
    return not any(f_status)

def main():
    anchor_point = None 
    start_cursor_x, start_cursor_y = 0, 0
    prev_x, prev_y = pyautogui.position()
    last_click_time = 0 
    
    # VARIABEL STATE (Kunci agar eksekusi hanya 1x)
    action_executed = False 
    
    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    ) as hands:
        while True:
            success, img = cap.read()
            if not success: continue 

            img = cv2.flip(img, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            
            # Variabel penampung teks status (Mencegah font tumpang tindih)
            display_text = ""
            text_color = (255, 255, 255)
            
            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    cx_pix = int((sum(hand_lms.landmark[id].x for id in PALM_IDS) / len(PALM_IDS)) * w_cam)
                    cy_pix = int((sum(hand_lms.landmark[id].y for id in PALM_IDS) / len(PALM_IDS)) * h_cam)

                    if anchor_point is None:
                        anchor_point = (cx_pix, cy_pix)
                        start_cursor_x, start_cursor_y = pyautogui.position()

                    f_status = get_fingers_status(hand_lms)
                    gripping = is_gripping(hand_lms)
                    current_time = time.time()

                 # --- LOGIKA DETEKSI IBU JARI (THUMB) BARU ---
                    thumb_tip = hand_lms.landmark[4]   # Bagian dari garis jempol
                    index_mcp = hand_lms.landmark[5]   # Titik atas pendeteksi sendi
                    wrist = hand_lms.landmark[0]       # Titik bawah pendeteksi sendi
                    pinky_mcp = hand_lms.landmark[17]  # Referensi area "dalam" telapak tangan

                    # Fungsi matematika untuk menentukan posisi titik terhadap garis 0 -> 5
                    # Menggunakan Cross Product vektor 2D
                    def get_side(point):
                        return (index_mcp.x - wrist.x) * (point.y - wrist.y) - (index_mcp.y - wrist.y) * (point.x - wrist.x)

                    # Menghitung posisi ujung jempol dan pangkal kelingking terhadap garis 0-5
                    side_thumb = get_side(thumb_tip)
                    side_pinky = get_side(pinky_mcp)

                    # Logikanya: Jika jempol melewati garis pendeteksi (0-5) ke arah dalam, 
                    # maka nilai posisinya akan berada di sisi yang sama dengan kelingking (pinky_mcp).
                    # Jika keduanya di sisi yang sama, hasil perkaliannya pasti positif (> 0).
                    thumb_closed = (side_thumb * side_pinky) > 0
                    
                    # Output variabel yang dibutuhkan oleh sisa program
                    thumb_open = not thumb_closed

                    # --- LOGIKA ONE-SHOT ACTION ---
                    
                    # 1. POSE SHIFT + N (4 Jari Berdiri, Ibu Jari Menekuk) -> ANGKA 4
                    if f_status[0] and f_status[1] and f_status[2] and f_status[3] and not thumb_open:
                        if not action_executed:
                            pyautogui.hotkey('shift', 'n')
                            action_executed = True 
                        display_text = "EXECUTED: SHIFT + N"
                        text_color = (27, 7, 207) # Biru
                    
                    # 2. POSE REFRESH (Telunjuk, Tengah, Manis Berdiri, Kelingking Tekuk) -> ANGKA 3
                    elif f_status[0] and f_status[1] and f_status[2] and not f_status[3]:
                        if not action_executed:
                            pyautogui.press('f5')
                            action_executed = True 
                        display_text = "EXECUTED: REFRESH (F5)"
                        text_color = (255, 255, 0) # Kuning
                        
                    # 3. POSE CLOSE (Telunjuk & Tengah Berdiri) -> ANGKA 2
                    elif f_status[0] and f_status[1] and not f_status[2] and not f_status[3]:
                        if not action_executed:
                            pyautogui.hotkey('alt', 'f4')
                            action_executed = True 
                        display_text = "EXECUTED: CLOSE (ALT+F4)"
                        text_color = (0, 0, 255) # Merah
                    
                    # 4. POSE SCROLL (Hanya Telunjuk Berdiri) -> ANGKA 1
                    elif f_status[0] and not f_status[1] and not f_status[2] and not f_status[3]:
                        if not action_executed:
                            pyautogui.scroll(-500) 
                            action_executed = True 
                        display_text = "EXECUTED: SCROLL"
                        text_color = (0, 255, 0) # Hijau
                    
                    # 5. POSE PAUSE / MENGEPAL
                    elif gripping:
                        action_executed = False 
                        display_text = "PAUSED / RESET ACTION"
                        text_color = (255, 255, 255) # Putih
                    
                    # 6. POSE DEFAULT / NEUTRAL (Termasuk saat semua 5 jari membuka)
                    else:
                        action_executed = False 

                    # --- PERGERAKAN KURSOR & KLIK ---
                    if not gripping:
                        dx = cx_pix - anchor_point[0]
                        dy = cy_pix - anchor_point[1]
                        scale = 2.5 

                        target_x = int(np.clip(start_cursor_x + (dx * scale), 0, screen_width))
                        target_y = int(np.clip(start_cursor_y + (dy * scale), 0, screen_height))

                        if math.hypot(target_x - prev_x, target_y - prev_y) > DEADZONE:
                            pyautogui.moveTo(target_x, target_y, duration=0)
                            prev_x, prev_y = target_x, target_y

                        # Logika Pinch (Klik Kiri)
                        index_tip = hand_lms.landmark[8]
                        thumb_tip_lm = hand_lms.landmark[4]
                        d = math.hypot(index_tip.x - thumb_tip_lm.x, index_tip.y - thumb_tip_lm.y) * w_cam
                        
                        if d < PINCH_START and (current_time - last_click_time) > 0.4:
                            pyautogui.click()
                            last_click_time = current_time

                    # Gambar teks status ke layar
                    if display_text:
                        cv2.putText(img, display_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

                    mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)
            else:
                anchor_point = None
                action_executed = False # Reset saat tangan hilang

            cv2.imshow("One-Shot Hand Control", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()