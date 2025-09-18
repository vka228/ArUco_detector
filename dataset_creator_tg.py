import cv2
import os
import time
import asyncio
from telethon import TelegramClient, events
import re

# Telegram API credentials (get from https://my.telegram.org)
API_ID = '26594365'  # Replace with your API ID
API_HASH = '5f5b7a66b2c4578fac6593bb028e17a2'  # Replace with your API hash
PHONE_NUMBER = '89104407043'  # Replace with your phone number with country code

# Define the base directory
base_dir = "calibration/datasets/aruco_test"
cam_dir = os.path.join(os.path.expanduser("~"), base_dir, "cam5")
os.makedirs(cam_dir, exist_ok=True)

# Initialize the camera
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    raise SystemExit(1)

# Set camera parameters
cap.set(cv2.CAP_PROP_BRIGHTNESS, 140)
cap.set(cv2.CAP_PROP_CONTRAST, 150)
cap.set(cv2.CAP_PROP_SATURATION, 100)
cap.set(cv2.CAP_PROP_SHARPNESS, 100)
cap.set(cv2.CAP_PROP_GAIN, 20)
cap.set(cv2.CAP_PROP_AUTO_WB, 0)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cap.set(cv2.CAP_PROP_FOCUS, -1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Camera resolution set to: {width}x{height}")

WINDOW_NAME = "Camera Feed"
cv2.namedWindow(WINDOW_NAME)

# Global variables for saving images
save_requested = False
distance_value = None
flash_until = 0.0
FLASH_DURATION = 0.6

# Create Telegram client
client = TelegramClient('aruco_capture_session', API_ID, API_HASH)


@client.on(events.NewMessage(chats='me'))  # Only monitor saved messages (messages to yourself)
async def handler(event):
    global save_requested, distance_value

    message_text = event.message.text
    print(f"Received message: {message_text}")

    # Look for distance pattern (e.g., "distance: 150", "dist 150", "150")
    match = re.search(r'(?:distance|dist)[:\s]*(\d+)', message_text, re.IGNORECASE)
    if not match:
        match = re.search(r'^(\d+)$', message_text)

    if match:
        distance_value = match.group(1)
        save_requested = True
        print(f"Distance detected: {distance_value}. Preparing to capture image...")
        await event.reply(f"Distance {distance_value} received. Capturing image...")
    else:
        print("No valid distance found in message")
        await event.reply("Please send a message with a distance value (e.g., 'distance: 150' or just '150')")


async def telegram_listener():
    await client.start(phone=PHONE_NUMBER)
    print("Telegram client started. Listening for messages...")
    await client.run_until_disconnected()


def capture_loop():
    global save_requested, distance_value, flash_until

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image.")
                break

            # If save is requested, save the current frame
            if save_requested and distance_value:
                save_requested = False
                filename = os.path.join(cam_dir, f"{distance_value}.png")
                ok = cv2.imwrite(filename, frame)
                if ok:
                    print(f"Saved image: {filename}")
                    flash_until = time.monotonic() + FLASH_DURATION
                else:
                    print(f"Error: Failed to write image: {filename}")
                distance_value = None

            # Draw flash indicator if needed
            now = time.monotonic()
            if now < flash_until:
                overlay = frame.copy()
                h, w = frame.shape[:2]
                cv2.rectangle(overlay, (0, 0), (w, h), (255, 255, 255), thickness=-1)
                alpha = 0.25  # flash transparency
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

                label = "SAVED IMG"
                cv2.putText(frame, label, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (80, 220, 80), 3, cv2.LINE_AA)

            display_frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)
            cv2.imshow(WINDOW_NAME, display_frame)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                print("User requested to stop.")
                break

    except KeyboardInterrupt:
        print("Image capture stopped by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released and script ended.")


async def main():
    # Run Telegram listener and capture loop concurrently
    await asyncio.gather(
        asyncio.to_thread(capture_loop),
        telegram_listener()
    )


if __name__ == "__main__":
    # Check if API credentials are set
    if API_ID == 'YOUR_API_ID' or API_HASH == 'YOUR_API_HASH' or PHONE_NUMBER == 'YOUR_PHONE_NUMBER':
        print("ERROR: Please set your Telegram API credentials in the code")
        print("1. Go to https://my.telegram.org")
        print("2. Create an application and get API_ID and API_HASH")
        print("3. Replace the placeholder values in the code with your credentials")
        exit(1)

    asyncio.run(main())