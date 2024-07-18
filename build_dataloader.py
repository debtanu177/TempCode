import torch

class AssignResult:
    def __init__(self, num_gts, gt_inds, max_overlaps, labels=None):
        """
        Args:
            num_gts (int): Number of ground truth boxes.
            gt_inds (Tensor): A tensor of shape (num_anchors,) indicating the index of the
                              assigned ground truth box for each anchor. -1 means unassigned,
                              0 means assigned to background, 1-based positive numbers mean
                              the index of the assigned ground truth box.
            max_overlaps (Tensor): A tensor of shape (num_anchors,) indicating the maximum
                                   IoU with the ground truth boxes for each anchor.
            labels (Tensor, optional): A tensor of shape (num_anchors,) indicating the labels
                                       of the assigned ground truth boxes. Defaults to None.
        """
        self.num_gts = num_gts
        self.gt_inds = gt_inds
        self.max_overlaps = max_overlaps
        self.labels = labels

    def add_gt_(self, gt_labels):
        """
        Add ground truth as assigned results.
        
        Args:
            gt_labels (Tensor): Labels of ground truth boxes.
        """
        assert self.gt_inds.max() <= 0
        self_inds = torch.arange(1, self.num_gts + 1, dtype=self.gt_inds.dtype, device=self.gt_inds.device)
        self.gt_inds = torch.cat([self_inds, self.gt_inds])
        self.max_overlaps = torch.cat([torch.ones(self.num_gts, dtype=self.max_overlaps.dtype, device=self.max_overlaps.device), self.max_overlaps])
        if self.labels is not None:
            self.labels = torch.cat([gt_labels, self.labels])

# Example usage:
num_gts = 3
gt_inds = torch.tensor([-1, 0, 1, 2, -1])
max_overlaps = torch.tensor([0.1, 0.3, 0.9, 0.8, 0.05])
labels = torch.tensor([1, 2, 0, 0, 1])

assign_result = AssignResult(num_gts, gt_inds, max_overlaps, labels)
print("Before adding GTs:")
print("gt_inds:", assign_result.gt_inds)
print("max_overlaps:", assign_result.max_overlaps)
print("labels:", assign_result.labels)

# Adding ground truth labels
gt_labels = torch.tensor([3, 4, 5])
assign_result.add_gt_(gt_labels)
print("After adding GTs:")
print("gt_inds:", assign_result.gt_inds)
print("max_overlaps:", assign_result.max_overlaps)
print("labels:", assign_result.labels)

















import torch

class BaseAssigner:
    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        """
        Assign ground truth boxes to anchors or proposals.

        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape (num_bboxes, 4).
            gt_bboxes (Tensor): Ground truth boxes, shape (num_gts, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth boxes to ignore, shape (num_ignored_gts, 4).
            gt_labels (Tensor, optional): Labels of ground truth boxes, shape (num_gts,).

        Returns:
            AssignResult: The assignment result.
        """
        raise NotImplementedError

class SimpleAssigner(BaseAssigner):
    def assign(self, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        """
        Simple implementation of the assign method for demonstration.

        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape (num_bboxes, 4).
            gt_bboxes (Tensor): Ground truth boxes, shape (num_gts, 4).
            gt_bboxes_ignore (Tensor, optional): Ground truth boxes to ignore, shape (num_ignored_gts, 4).
            gt_labels (Tensor, optional): Labels of ground truth boxes, shape (num_gts,).

        Returns:
            AssignResult: The assignment result.
        """
        num_bboxes = bboxes.size(0)
        num_gts = gt_bboxes.size(0)

        # Initialize assignment results
        assigned_gt_inds = bboxes.new_full((num_bboxes,), -1, dtype=torch.long)
        max_overlaps = bboxes.new_zeros((num_bboxes,))
        assigned_labels = bboxes.new_full((num_bboxes,), -1, dtype=torch.long)

        if num_gts == 0:
            # No ground truth, assign all to background
            assigned_gt_inds[:] = 0
            return AssignResult(num_gts, assigned_gt_inds, max_overlaps, assigned_labels)

        # Compute IoU between bboxes and gt_bboxes
        overlaps = self.compute_overlaps(bboxes, gt_bboxes)

        # Assign each bbox to the ground truth with highest IoU
        max_overlaps, argmax_overlaps = overlaps.max(dim=1)
        assigned_gt_inds[max_overlaps > 0] = argmax_overlaps[max_overlaps > 0] + 1

        if gt_labels is not None:
            assigned_labels[max_overlaps > 0] = gt_labels[argmax_overlaps[max_overlaps > 0]]

        return AssignResult(num_gts, assigned_gt_inds, max_overlaps, assigned_labels)

    def compute_overlaps(self, bboxes, gt_bboxes):
        """
        Compute IoU between bounding boxes and ground truth boxes.

        Args:
            bboxes (Tensor): Bounding boxes, shape (num_bboxes, 4).
            gt_bboxes (Tensor): Ground truth boxes, shape (num_gts, 4).

        Returns:
            Tensor: IoU between each pair of bboxes and gt_bboxes, shape (num_bboxes, num_gts).
        """
        num_bboxes = bboxes.size(0)
        num_gts = gt_bboxes.size(0)

        overlaps = torch.zeros((num_bboxes, num_gts), dtype=torch.float)

        for i in range(num_bboxes):
            for j in range(num_gts):
                overlaps[i, j] = self.iou(bboxes[i], gt_bboxes[j])

        return overlaps

    def iou(self, bbox1, bbox2):
        """
        Compute IoU between two bounding boxes.

        Args:
            bbox1 (Tensor): Bounding box, shape (4,).
            bbox2 (Tensor): Bounding box, shape (4,).

        Returns:
            float: IoU between bbox1 and bbox2.
        """
        x1, y1, x2, y2 = bbox1
        xx1, yy1, xx2, yy2 = bbox2

        inter_x1 = max(x1, xx1)
        inter_y1 = max(y1, yy1)
        inter_x2 = min(x2, xx2)
        inter_y2 = min(y2, yy2)

        inter_area = max(0, inter_x2 - inter_x1 + 1) * max(0, inter_y2 - inter_y1 + 1)
        bbox1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
        bbox2_area = (xx2 - xx1 + 1) * (yy2 - yy2 + 1)

        iou = inter_area / float(bbox1_area + bbox2_area - inter_area)
        return iou

class AssignResult:
    def __init__(self, num_gts, gt_inds, max_overlaps, labels=None):
        """
        Args:
            num_gts (int): Number of ground truth boxes.
            gt_inds (Tensor): A tensor of shape (num_anchors,) indicating the index of the
                              assigned ground truth box for each anchor. -1 means unassigned,
                              0 means assigned to background, 1-based positive numbers mean
                              the index of the assigned ground truth box.
            max_overlaps (Tensor): A tensor of shape (num_anchors,) indicating the maximum
                                   IoU with the ground truth boxes for each anchor.
            labels (Tensor, optional): A tensor of shape (num_anchors,) indicating the labels
                                       of the assigned ground truth boxes. Defaults to None.
        """
        self.num_gts = num_gts
        self.gt_inds = gt_inds
        self.max_overlaps = max_overlaps
        self.labels = labels

    def add_gt_(self, gt_labels):
        """
        Add ground truth as assigned results.

        Args:
            gt_labels (Tensor): Labels of ground truth boxes.
        """
        assert self.gt_inds.max() <= 0
        self_inds = torch.arange(1, self.num_gts + 1, dtype=self.gt_inds.dtype, device=self.gt_inds.device)
        self.gt_inds = torch.cat([self_inds, self.gt_inds])
        self.max_overlaps = torch.cat([torch.ones(self.num_gts, dtype=self.max_overlaps.dtype, device=self.max_overlaps.device), self.max_overlaps])
        if self.labels is not None:
            self.labels = torch.cat([gt_labels, self.labels])

# Example usage:
bboxes = torch.tensor([[50, 50, 100, 100], [30, 30, 70, 70]], dtype=torch.float)
gt_bboxes = torch.tensor([[40, 40, 80, 80], [60, 60, 90, 90]], dtype=torch.float)
gt_labels = torch.tensor([1, 2], dtype=torch.long)

assigner = SimpleAssigner()
assign_result = assigner.assign(bboxes, gt_bboxes, gt_labels=gt_labels)

print("Assigned GT indices:", assign_result.gt_inds)
print("Max overlaps:", assign_result.max_overlaps)
print("Assigned labels:", assign_result.labels)
