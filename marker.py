import qrcode

# QR 코드 데이터
data = "Hello, this is a marker!"

# QR 코드 생성
qr = qrcode.QRCode(version=1, box_size=10, border=5)
qr.add_data(data)
qr.make(fit=True)

# 이미지 생성
img = qr.make_image(fill='black', back_color='white')

# 이미지 저장
img.save("marker.png")