import cv2
import numpy as np
import matplotlib.pyplot as plt

# تبدیل تصویر RGB به HSI
def rgb_to_hsi(image):
    with np.errstate(divide='ignore', invalid='ignore'):
        image = np.float32(image) / 255  # نرمال‌سازی تصویر
        B, G, R = cv2.split(image)
        
        num = 0.5 * ((R - G) + (R - B))
        denom = np.sqrt((R - G)**2 + (R - B)*(G - B))
        H = np.arccos(num / (denom + 1e-5))  # اجتناب از تقسیم بر صفر
        
        H[B > G] = 2 * np.pi - H[B > G]
        H = H / (2 * np.pi)  # نرمال‌سازی بین 0 و 1
        
        S = 1 - (3 / (R + G + B + 1e-5) * np.minimum(R, np.minimum(G, B)))
        I = (R + G + B) / 3
        
        return cv2.merge([H, S, I])

# بارگذاری تصویر
image = cv2.imread('E:/saffronImageProcess/OIP.jpg')

# پیش‌پردازش تصویر: افزایش کنتراست
image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(image_lab)
l = cv2.equalizeHist(l)
image_eq = cv2.merge((l, a, b))
image_eq = cv2.cvtColor(image_eq, cv2.COLOR_LAB2BGR)

# تبدیل تصویر به فضای HSI
hsi_image = rgb_to_hsi(image_eq)
H, S, I = cv2.split(hsi_image)

# بهبود تشخیص رنگ بنفش و قرمز (رنگ گل زعفران)
lower_bound = np.array([0.6, 0.2, 0.2])  # محدوده جدید رنگ بنفش در HSI
upper_bound = np.array([0.9, 1.0, 1.0])

# ماسک رنگ بنفش
mask = cv2.inRange(hsi_image, lower_bound, upper_bound)

# فیلتر حذف نویز
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel)

# فیلتر نویز با استفاده از فیلتر Bilateral
image_filtered = cv2.bilateralFilter(image_eq, 9, 75, 75)

# یافتن کانتورهای نواحی گل
contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# رسم کانتورها روی تصویر اصلی و نمایش مشخصات
output = image.copy()
font = cv2.FONT_HERSHEY_SIMPLEX

for i, cnt in enumerate(contours):
    area = cv2.contourArea(cnt)
    if area > 150:  # حذف نواحی خیلی کوچک
        # رسم مستطیل اطراف کانتور
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # محاسبه مرکز هر کانتور
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(output, (cX, cY), 5, (0, 0, 255), -1)  # رسم مرکز
            
            # متن شامل مشخصات گل: مرکز و ابعاد مستطیل
            text = f"Flower {i}: (x={x}, y={y}), w={w}, h={h}, center=({cX}, {cY})"
            
            # نوشتن متن روی تصویر
            text_position = (x, y - 10 if y - 10 > 10 else y + 10)
            cv2.putText(output, text, text_position, font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            
            # چاپ مشخصات در ترمینال
            print(text)

# نمایش تصویر نهایی
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title('Detected Saffron Flowers')
plt.axis('off')
plt.show()

# ذخیره تصویر نهایی در فایل
cv2.imwrite('E:/saffronImageProcess/Detected_Saffron_Flowers.jpg', output)
