import cv2
import argparse
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='best.pt')
    parser.add_argument('--source', type=str, default='test.mp4')
    parser.add_argument('--output', type=str, default='result.mp4')
    args = parser.parse_args()

    model = YOLO(args.model)
    names = model.names
    cap = cv2.VideoCapture(args.source)
    width, height, fps = int(cap.get(3)), int(cap.get(4)), int(cap.get(5))
    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        results = model(frame, verbose=False)
        result = results[0]

        if result.boxes is not None:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                label = f"{names[cls_id]} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if result.keypoints is not None:
            for kpt in result.keypoints.xy.cpu().numpy():
                for x, y in kpt:
                    cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)
        out.write(frame)
    cap.release()
    out.release()
    print(f"完成！结果保存为: {args.output}")

if __name__ == '__main__':
    main()
