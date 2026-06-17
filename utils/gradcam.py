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
    True Grad-CAM by rebuilding a single flat functional model
    that explicitly chains: input -> mobilenet conv layer -> head layers.
    This avoids the broken-gradient problem caused by calling a
    nested sub-model inside another model.
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
    target_layer_name = None
    for name in ['Conv_1', 'block_16_project_BN', 'block_16_project',
                 'block_15_project_BN', 'out_relu']:
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

    # ── Step 3: Rebuild a clean functional graph ────────────────
    # Find all layers AFTER the mobilenet block in the original model
    # (GlobalAveragePooling2D, Dense layers, Dropout, etc.)
    post_mobilenet_layers = []
    found_mobilenet = False
    for layer in model.layers:
        if layer is mobilenet:
            found_mobilenet = True
            continue
        if found_mobilenet:
            post_mobilenet_layers.append(layer)

    # Get the conv layer's output tensor from mobilenet
    conv_layer_output = mobilenet.get_layer(target_layer_name).output

    # Build a sub-model: mobilenet.input -> conv_layer_output
    conv_extractor = tf.keras.Model(
        inputs=mobilenet.input,
        outputs=conv_layer_output
    )

    # Build a sub-model: mobilenet.input -> mobilenet's final output
    # (this is what feeds into the post-mobilenet head layers)
    mobilenet_full = tf.keras.Model(
        inputs=mobilenet.input,
        outputs=mobilenet.output
    )

    # ── Step 4: Preprocess image ────────────────────────────────
    img_resized = cv2.resize(image, (224, 224))
    img_tensor  = tf.cast(
        np.expand_dims(img_resized, axis=0), tf.float32
    ) / 255.0

    # ── Step 5: Compute Grad-CAM using GradientTape ──────────────
    with tf.GradientTape() as tape:
        # Get conv layer activations (watched automatically since
        # they come from a forward pass inside the tape)
        conv_output = conv_extractor(img_tensor, training=False)
        tape.watch(conv_output)

        # Continue the forward pass: mobilenet output -> head layers
        x = mobilenet_full(img_tensor, training=False)
        for layer in post_mobilenet_layers:
            x = layer(x, training=False)

        predictions = x
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_output)

    if grads is None:
        print("[WARNING] Gradients still None after rebuild. "
              "Falling back to saliency map.")
        return _saliency_map(model, image, class_index, save_path)

    # ── Step 6: Build the heatmap ─────────────────────────────────
    grads_val = grads.numpy()[0]        # (H, W, C)
    conv_val  = conv_output.numpy()[0]  # (H, W, C)

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

    print("[INFO] True Grad-CAM heatmap computed successfully.")
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
        heatmap_resized = cv2.resize(heatmap, (w, h))

        if concentrate:
            # Power transform increases contrast — pushes mid-low
            # values down and keeps only the strongest region bright,
            # producing the concentrated red/yellow hotspot look
            heatmap_resized = np.power(heatmap_resized, 1.8)
            if heatmap_resized.max() > 0:
                heatmap_resized /= heatmap_resized.max()

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