from ultralytics import YOLO
import cv2

model = YOLO('models/best.pt')
print(model.names)

img = cv2.imread('test_image/img2.jpg')

result = model.predict(img)
detect = result[0].plot()

# Count Calories
calories_data = {
    'egg'     : 155,
    'meat'    : 143,
    'porridge': 50,
    'rice'    : 130,
    'soup'    : 164
}

detected_foods = ['egg','meat','porridge','rice','soup']

total_calories = sum(calories_data.get(food, 0) for food in detected_foods)
print('Total Calories:', total_calories)

cv2.imshow('detection',detect)
cv2.waitKey(0)