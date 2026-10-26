def count_intersection_union(pred, target, num_classes):
    intersections = [0] * num_classes
    unions = [0] * num_classes

    for c in range(num_classes):
        intersections[c] = ((pred == c) & (target == c)).sum().item()
        unions[c] = ((pred == c) | (target == c)).sum().item()

    return intersections, unions

def iou_from_counts(intersections, unions):
    iou_list = []
    for inter, union in zip(intersections, unions):
        iou_list.append(None if union == 0 else inter/union)
    return iou_list

def mean_iou(iou_list):
    valid = []
    for x in iou_list:
        if x is not None:
            valid.append(x)
    return sum(valid) / len(valid) if valid else 0.0