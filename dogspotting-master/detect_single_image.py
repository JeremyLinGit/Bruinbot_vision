from imageai.Detection import ObjectDetection

## Set up detector
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()

# Load model
model_path = 'resnet50_coco_best_v2.1.0.h5'

detector.setModelPath(model_path)
detector.loadModel()

# Set custom object detection for dogs and cats
# For more options, see: https://github.com/OlafenwaMoses/ImageAI/tree/master/imageai/Detection#---custom-object-detection
custom_objects = detector.CustomObjects(dog=True, cat=True)

## Detect on image
input_path = 'cad2.jpeg'
output_path = 'result4.png'

# Detect
detections = detector.detectCustomObjectsFromImage(input_image=input_path, output_image_path=output_path, custom_objects=custom_objects, minimum_percentage_probability=45)

