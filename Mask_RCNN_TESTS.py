
import cv2
import model as modellib
import coco


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

COCO_MODEL_PATH = "dependencies/pretreinedmodels/mask_rcnn_coco.h5"

# Directory to save logs and trained model
MODEL_DIR = "logs"

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

# COCO Class names
class_names = ['BG', 'Person', 'Bicycle', 'Car', 'Motorcycle', 'Airplane',
               'Bus', 'Train', 'Truck', 'Boat', 'Traffic Light',
               'Fire Hydrant', 'Stop Sign', 'Parking Meter', 'Bench', 'Bird',
               'Cat', 'Dog', 'Horse', 'Sheep', 'Cow', 'Elephant', 'Bear',
               'Zebra', 'Giraffe', 'Backpack', 'Umbrella', 'Handbag', 'Tie',
               'Suitcase', 'Frisbee', 'Skis', 'Snowboard', 'Sports Ball',
               'Kite', 'Baseball Bat', 'Baseball Glove', 'Skateboard',
               'Surfboard', 'Tennis Racket', 'Bottle', 'Wine Glass', 'Cup',
               'Fork', 'Knife', 'Spoon', 'Bowl', 'Banana', 'Spple',
               'Sandwich', 'Orange', 'Broccoli', 'Carrot', 'Hot Dog', 'Pizza',
               'Donut', 'Cake', 'Chair', 'Couch', 'Potted Plant', 'Bed',
               'Dining Table', 'Toilet', 'TV', 'Laptop', 'Mouse', 'Remote',
               'Keyboard', 'Cell Phone', 'Microwave', 'Oven', 'Toaster',
               'Sink', 'Refrigerator', 'Book', 'Clock', 'Vase', 'Scissors',
               'Teddy Bear', 'Hair Drier', 'Toothbrush']


image = cv2.imread("images/2007_000480.jpg")

# Run detection
results = model.detect([image], verbose=1)

# Visualize results
r = results[0]



for i in range(r['rois'].shape[0]):
    y1, x1, y2, x2 = r['rois'][i]
    mask_img = cv2.rectangle(image, (x1,y1), (x2,y2), 255, 1)
    cv2.putText(mask_img, class_names[r['class_ids'][i]], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,55,255), 1)



while(1):
        cv2.imshow('image',mask_img)
        k = cv2.waitKey(33)
        if k==27:   
            break
            cv2.destroyAllWindows()
        elif k==-1:  
            continue


cv2.imwrite('result.jpg',mask_img)


