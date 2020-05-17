
def select_humans(predictions):

    instances = predictions['instances']
    n = len(instances)
    classes = instances.pred_classes.tolist()
    bboxes = instances.pred_boxes.tensor.tolist()

    human_boxes = [bboxes[i] for i in range(n) if classes[i] == 0]

    if human_boxes:
        return human_boxes
    else:
        return []
