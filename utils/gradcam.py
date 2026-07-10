"""
Grad-CAM Heatmap Generator — Final Working Version
Rebuilds a clean functional model so gradients flow correctly
through the nested MobileNetV2 sub-model.
"""

import cv2
import numpy as np
import tensorflow as tf


def generate_gradcam_heatmap(model, image, class_index, save_path):
    try:
        return _real_gradcam(model, image, class_index, save_path)
    except Exception as e:
        print(f"[WARNING] Grad-CAM failed: {e}")
        import traceback
        traceback.print_exc()
        return _gradcam_fallback(model, image, class_index, save_path)


def _real_gradcam(model, image, class_index, save_path):
    """
    True Grad-CAM targeting a higher-resolution intermediate layer inside
    the nested MobileNetV2 sub-model (14x14), instead of its final 7x7
    output. The 7x7 grid is too coarse to separate cheeks/forehead/chin/
    jawline, and block_13_expand_relu is the deepest 14x14 layer that
    still falls inside the fine-tuned (unfrozen) portion of the network
    (see training/train.py: base.layers[-40:] are trainable), so its
    activations are actually shaped by the acne-classification task.
    """
    # ── Step 1: Locate the MobileNetV2 sub-model ──────────────
    mobilenet = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            mobilenet = layer
            break

    if mobilenet is None:
        print("[INFO] No nested sub-model found, using flat search.")
        return _flat_model_gradcam(model, image, class_index, save_path)

    # ── Step 2: Find best conv layer ───────────────────────────
    # Preference order: highest-resolution layer still inside the
    # fine-tuned range, falling back to coarser/frozen layers only
    # if the preferred ones aren't present in this architecture.
    target_layer_name = None
    for name in ['block_13_expand_relu', 'block_13_expand_BN',
                 'block_12_project_BN', 'block_16_project_BN',
                 'Conv_1', 'out_relu']:
        try:
            mobilenet.get_layer(name)
            target_layer_name = name
            break
        except ValueError:
            continue

    if target_layer_name is None:
        for layer in reversed(mobilenet.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                target_layer_name = layer.name
                break

    if target_layer_name is None:
        print("[WARNING] No conv layer found in sub-model.")
        return _gradcam_fallback(model, image, class_index, save_path)

    print(f"[INFO] Grad-CAM target layer: '{target_layer_name}' "
          f"in '{mobilenet.name}'")

    # ── Step 3: Two outputs from the SAME sub-model graph ────────
    # Keras 3 won't let you reach into a nested sub-model's internal
    # tensor from the *outer* model's graph (that's the "KerasTensor
    # cannot be used..." / graph KeyError you get if you try). But
    # asking the sub-model itself for two of its own tensors is fine
    # -- it's one self-contained functional graph.
    mobilenet_multi = tf.keras.Model(
        inputs=mobilenet.input,
        outputs=[mobilenet.get_layer(target_layer_name).output, mobilenet.output],
    )

    # Head layers (GlobalAveragePooling2D, Dense, Dropout, ...) that
    # come after the mobilenet block in the outer model.
    post_mobilenet_layers = []
    found_mobilenet = False
    for layer in model.layers:
        if layer is mobilenet:
            found_mobilenet = True
            continue
        if found_mobilenet:
            post_mobilenet_layers.append(layer)

    # ── Step 4: Preprocess image ────────────────────────────────
    img_resized = cv2.resize(image, (224, 224))
    img_tensor  = tf.cast(
        np.expand_dims(img_resized, axis=0), tf.float32
    ) / 255.0

    # ── Step 5: Compute Grad-CAM using GradientTape ──────────────
    # Everything from here on runs eagerly (real tensors, not
    # symbolic graph-building), so replaying the head layers by hand
    # and bypassing the final softmax for raw logits is safe.
    with tf.GradientTape() as tape:
        conv_output, mob_out = mobilenet_multi(img_tensor, training=False)
        tape.watch(conv_output)

        x = mob_out
        for layer in post_mobilenet_layers[:-1]:
            x = layer(x, training=False)

        # Last layer is the softmax classifier head. Softmax saturates
        # near-confident predictions and washes out the gradient
        # signal, so use the raw logit instead.
        last_layer = post_mobilenet_layers[-1]
        if getattr(last_layer, 'activation', None) is not None and hasattr(last_layer, 'kernel'):
            logits = tf.matmul(x, last_layer.kernel) + last_layer.bias
        else:
            logits = last_layer(x, training=False)

        loss = logits[:, class_index]

    grads = tape.gradient(loss, conv_output)

    if grads is None:
        print("[WARNING] Gradients still None. Falling back to saliency map.")
        return _saliency_map(model, image, class_index, save_path)

    # ── Step 7: Build the heatmap ─────────────────────────────────
    grads_val = grads.numpy()[0]       # (H, W, C)
    conv_val  = conv_output.numpy()[0] # (H, W, C)

    # Global average pool gradients across spatial dimensions
    weights = np.mean(grads_val, axis=(0, 1))  # (C,)

    # Weighted combination of feature maps
    heatmap = np.zeros(conv_val.shape[:2], dtype=np.float32)
    for i, w in enumerate(weights):
        heatmap += w * conv_val[:, :, i]

    # ReLU: keep only positive contributions
    heatmap = np.maximum(heatmap, 0)

    # Normalise
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    print(f"[INFO] True Grad-CAM heatmap computed successfully "
          f"({conv_val.shape[0]}x{conv_val.shape[1]} grid).")
    return _save_heatmap(heatmap, image, save_path, concentrate=True)


def _flat_model_gradcam(model, image, class_index, save_path):
    """Used when the model has no nested sub-model layer."""
    target_layer_name = None
    for name in ['Conv_1', 'block_16_project_BN', 'out_relu']:
        try:
            model.get_layer(name)
            target_layer_name = name
            break
        except ValueError:
            continue

    if target_layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                target_layer_name = layer.name
                break

    if target_layer_name is None:
        return _saliency_map(model, image, class_index, save_path)

    img_resized = cv2.resize(image, (224, 224))
    img_tensor  = tf.cast(
        np.expand_dims(img_resized, axis=0), tf.float32
    ) / 255.0

    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[model.get_layer(target_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_tensor, training=False)
        tape.watch(conv_output)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_output)
    if grads is None:
        return _saliency_map(model, image, class_index, save_path)

    grads_val = grads.numpy()[0]
    conv_val  = conv_output.numpy()[0]
    weights   = np.mean(grads_val, axis=(0, 1))
    heatmap   = np.zeros(conv_val.shape[:2], dtype=np.float32)
    for i, w in enumerate(weights):
        heatmap += w * conv_val[:, :, i]
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    return _save_heatmap(heatmap, image, save_path, concentrate=True)


def _saliency_map(model, image, class_index, save_path):
    """Input-gradient saliency — only used if true Grad-CAM is impossible."""
    try:
        print("[INFO] Using saliency map (last resort before radial).")
        img_resized = cv2.resize(image, (224, 224))
        tensor = tf.Variable(
            tf.cast(np.expand_dims(img_resized, axis=0), tf.float32) / 255.0
        )

        with tf.GradientTape() as tape:
            tape.watch(tensor)
            preds = model(tensor, training=False)
            loss  = preds[:, class_index]

        grads = tape.gradient(loss, tensor)
        if grads is None:
            return _gradcam_fallback(model, image, class_index, save_path)

        saliency = tf.reduce_max(tf.abs(grads), axis=-1)[0].numpy()
        saliency = np.maximum(saliency, 0)
        if saliency.max() > 0:
            saliency /= saliency.max()
        saliency = cv2.GaussianBlur(saliency.astype(np.float32), (21, 21), 0)

        return _save_heatmap(saliency, image, save_path, concentrate=True)

    except Exception as e:
        print(f"[WARNING] Saliency map failed: {e}")
        return _gradcam_fallback(model, image, class_index, save_path)


def _save_heatmap(heatmap, image, save_path, concentrate=False):
    """
    Resize, optionally sharpen contrast to concentrate the hotspot
    (like Image 1 reference), colourise and overlay.
    """
    try:
        h, w = image.shape[:2]
        # Cubic interpolation instead of the default bilinear avoids the
        # hard-edged diamond/rhombus artifact you get from upscaling a
        # coarse grid (7x7-14x14) straight to full image resolution.
        heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
        heatmap_resized = np.clip(heatmap_resized, 0, None)

        if concentrate:
            # Power transform increases contrast — pushes mid-low
            # values down and keeps the strongest region brightest,
            # producing a concentrated red/yellow hotspot look. Kept
            # gentle (1.2) so it no longer collapses the whole map
            # down to one blocky cell.
            heatmap_resized = np.power(heatmap_resized, 1.2)
            if heatmap_resized.max() > 0:
                heatmap_resized /= heatmap_resized.max()

        # Smooth away the residual grid-cell blockiness from the low
        # native resolution of the conv feature map.
        blur_k = max(3, (min(h, w) // 40) | 1)  # odd kernel, scales with image size
        heatmap_resized = cv2.GaussianBlur(heatmap_resized, (blur_k, blur_k), 0)

        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(img_bgr, 0.55, colored, 0.45, 0)

        cv2.imwrite(save_path, overlay)
        print(f"[INFO] Grad-CAM saved: {save_path}")
        return True

    except Exception as e:
        print(f"[WARNING] Heatmap save failed: {e}")
        return False


def _gradcam_fallback(model, image, class_index, save_path):
    """Last resort: confidence-weighted radial heatmap on face centre."""
    try:
        print("[INFO] Using radial fallback heatmap.")
        img_resized = cv2.resize(image, (224, 224))
        tensor = tf.cast(
            np.expand_dims(img_resized, axis=0), tf.float32
        ) / 255.0

        preds      = model(tensor, training=False).numpy()
        confidence = float(preds[0][class_index])

        h, w   = image.shape[:2]
        cx, cy = w // 2, int(h * 0.42)
        Y, X   = np.ogrid[:h, :w]
        radius = min(h, w) * 0.32
        dist   = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        heatmap = np.clip(1.0 - dist / radius, 0, 1) ** 2 * confidence
        heatmap = cv2.GaussianBlur(heatmap.astype(np.float32), (31, 31), 0)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        return _save_heatmap(heatmap, image, save_path, concentrate=False)

    except Exception as e:
        print(f"[WARNING] Radial fallback failed: {e}")
        return False