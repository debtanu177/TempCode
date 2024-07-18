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
