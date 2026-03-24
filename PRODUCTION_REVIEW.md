# Production Review

## Current Position

ระบบตอนนี้อยู่ในสถานะ production-oriented prototype มากกว่าระบบ production ที่พิสูจน์แล้ว

จุดที่ทำได้แล้ว:

1. โค้ดถูกแยกเป็น package จริงและลดความเสี่ยงจาก monolith เดียว
2. มีสถานะผลลัพธ์ `success`, `low_confidence`, `failed` สำหรับ downstream integration
3. detector ใช้ contour fallback ร่วมกับ YOLO-World prompt cascade เพื่อลด latency เฉลี่ย
4. evaluation รองรับ exact match, CER, IoU และ slice leaderboard
5. dependencies ถูก pin เวอร์ชันเพื่อให้ deploy ซ้ำได้ง่ายขึ้น

## Highest-Priority Risks

1. Generalization risk
   ระบบยังถูก verify กับ sample จำนวนน้อยมาก จึงยังอ้าง performance บนภาพจริงทุกเงื่อนไขไม่ได้
2. Model choice risk
   YOLO-World ยังเป็น generic detector ไม่ใช่ Thai plate detector โดยตรง จึงมีโอกาส false positive และ false negative ใน distribution ใหม่
3. OCR drift risk
   EasyOCR บนภาษาไทยยังอ่อนไหวกับ blur, glare, skew, และตัวอักษรใกล้เคียง
4. Runtime environment risk
   ตอนนี้ใช้ Python 3.14 ใน `.venv` ซึ่งทำงานได้กับเครื่องนี้ แต่ ecosystem support ยังไม่แข็งแรงเท่า Python 3.11 หรือ 3.12
5. Benchmark coverage risk
   ยังไม่มี benchmark ที่ครอบคลุมกลางวัน/กลางคืน/ระยะ/มุมกล้อง/รถหลายคัน/ป้ายบังบางส่วน

## Recommended Priority Order

1. สร้าง benchmark dataset จริงอย่างน้อยหลายร้อยภาพ พร้อม slice labels ตาม template
2. เก็บ baseline metrics แยกตาม `view`, `lighting`, `distance_bucket`, `occlusion`, `scene`
3. วัด false positive rate บนภาพที่ไม่มีป้ายหรือไม่มีรถ
4. พิจารณา train หรือ fine-tune detector เฉพาะงาน Thai plate detection
5. เพิ่ม regression suite สำหรับ output contract และ evaluation reports
6. ย้าย runtime target ไป Python 3.11 หรือ 3.12 สำหรับ deploy จริง

## Exit Criteria Before Real Production

1. มี validation set และ test set แยกชัดเจน
2. `success_rate` และ `combined_exact_accuracy` ผ่าน threshold ที่ธุรกิจกำหนดในทุก slice สำคัญ
3. `low_confidence` ถูก route ไป human review ได้จริง
4. มี monitoring สำหรับสัดส่วน `failed` และ `low_confidence`
5. มี model/version pinning และ rollout plan ย้อนกลับได้

## Suggested Operational Policy

1. ใช้เฉพาะ `success` สำหรับ auto-approve
2. ส่ง `low_confidence` เข้า manual review queue
3. treat `failed` เป็น no-read และเก็บภาพสำหรับ error analysis
4. เก็บ evaluation artifacts ทุก batch release เพื่อเปรียบเทียบ regression
