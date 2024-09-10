import os

# ระบุ path ของ video folder
video_folder = '../Model/input/data/videos_160/other'

# ลูปผ่านไฟล์ทั้งหมดในโฟลเดอร์
for filename in os.listdir(video_folder):
    if filename.endswith("mp4"):  # ตรวจสอบเฉพาะไฟล์ที่ลงท้ายด้วย mp4
        new_filename = filename.replace("mp4", ".mp4")  # เปลี่ยนชื่อไฟล์โดยเพิ่มจุดก่อน mp4
        # สร้าง path ใหม่สำหรับการเปลี่ยนชื่อ
        old_file = os.path.join(video_folder, filename)
        new_file = os.path.join(video_folder, new_filename)
        os.rename(old_file, new_file)  # เปลี่ยนชื่อไฟล์
        print(f'Renamed: {filename} -> {new_filename}')
