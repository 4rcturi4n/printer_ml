import torch
import torch.nn.functional as F


class VideoGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None
        self.handles = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inputs, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.handles.append(
            self.target_layer.register_forward_hook(forward_hook)
        )
        self.handles.append(
            self.target_layer.register_full_backward_hook(backward_hook)
        )

    def remove_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def generate(self, x):
        """
        x: [B, C, T, H, W]
        returns:
            cam: [B, T, H, W]
            output: model output
        """
        self.model.zero_grad(set_to_none=True)

        output = self.model(x)

        if output.ndim == 2 and output.shape[1] == 1:
            score = output[:, 0]
        elif output.ndim == 1:
            score = output
        else:
            raise ValueError(
                f"Expected regression output shape [B] or [B,1], got {tuple(output.shape)}"
            )

        score.sum().backward()

        if self.activations is None:
            raise RuntimeError("Activations not captured. Check target_layer.")
        if self.gradients is None:
            raise RuntimeError("Gradients not captured. Check target_layer.")

        activations = self.activations   # [B, C, t, h, w]
        gradients = self.gradients       # [B, C, t, h, w]

        weights = gradients.mean(dim=(2, 3, 4), keepdim=True)   # [B, C, 1, 1, 1]
        cam = (weights * activations).sum(dim=1)                # [B, t, h, w]
        #cam = F.relu(cam)

        cam = cam.unsqueeze(1)  # [B,1,t,h,w]
        cam = F.interpolate(
            cam,
            size=(x.shape[2], x.shape[3], x.shape[4]),
            mode="trilinear",
            align_corners=False,
        )
        cam = cam.squeeze(1)    # [B,T,H,W]

        cam_list = []
        for i in range(cam.shape[0]):
            c = cam[i]
            c = c - c.min()
            c = c / (c.max() + 1e-8)
            cam_list.append(c)

        cam = torch.stack(cam_list, dim=0)
        
        print("Input shape:", x.shape)
        print("Activation shape:", self.activations.shape)
        frame_strength = cam[0].mean(dim=(1, 2)).detach().cpu().numpy()
        print("frame strength:", frame_strength)
        
        return cam.detach(), output.detach()