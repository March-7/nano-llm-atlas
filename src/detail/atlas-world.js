import { basis, cameraMap, clamp } from './tensor-paint.js';

export const overviewYaw = 0.34;
export const overviewPitch = 0.22;
const frontal = basis(overviewYaw, overviewPitch);

export function worldInsets(width, height) {
  return {
    side: Math.min(width * 0.055, 38),
    top: Math.min(clamp(height * 0.18, 110, 135), height * 0.33),
    bottom: Math.min(clamp(height * 0.04, 20, 34), height * 0.1),
  };
}

/** A single camera rotates every tensor against the same vertical axis. The
 * comparison positions are expressed in its image plane, in atlas units. */
export function arrangeWorld(entries, width, height) {
  const { side, top, bottom } = worldInsets(width, height);
  const columns = Math.min(entries.length, width >= 680 ? 3 : 2);
  if (!columns) return new Map();
  const rows = Math.ceil(entries.length / columns);
  const slotWidth = Math.max(1, width - 2 * side) / columns;
  const slotHeight = Math.max(1, height - top - bottom) / rows;
  const labelSpace = Math.min(36, slotHeight * 0.25);
  const gap = Math.min(24, slotHeight * 0.1);
  return new Map(
    entries.map((entry, index) => {
      const { bounds, layers } = entry.layout;
      const localCenter = [0, 1, 2].map((i) => (bounds[i] + bounds[i + 3]) / 2);
      // A vocabulary table can protrude far to one side of the decoder tower.
      // Center the label and orbit over the repeated body, retaining both tables.
      if (layers.length) {
        for (const i of [0, 2]) {
          const low = Math.min(...layers.map((layer) => layer.bounds[i]));
          const high = Math.max(...layers.map((layer) => layer.bounds[i + 3]));
          localCenter[i] = (low + high) / 2;
        }
      }
      const radius = localCenter.map((v, i) => Math.max(v - bounds[i], bounds[i + 3] - v));
      const horizontal = 2 * Math.hypot(radius[0], radius[2]);
      const vertical = 2 * Math.hypot(...radius);
      // Maxima of |right·extent| and |down·extent| across yaw/pitch. Scale is
      // constant during the turn; adjacent footprints remain disjoint at any zoom.
      const unit = Math.max(
        1e-12,
        Math.min(
          (slotWidth * 0.88) / Math.max(1, horizontal),
          (slotHeight - labelSpace - gap) / Math.max(1, vertical),
        ),
      );
      const slot = [
        side + ((index % columns) + 0.5) * slotWidth - width / 2,
        top + (Math.floor(index / columns) + 0.5) * slotHeight + (labelSpace - gap) / 2 - height / 2,
      ];
      return [entry.id, { ...entry, localCenter, unit, slot }];
    }),
  );
}

export function modelOrigin(model, axes = frontal) {
  return axes.right.map((r, i) => r * model.slot[0] + axes.down[i] * model.slot[1]);
}

export function toWorld(model, local, axes = frontal) {
  const origin = modelOrigin(model, axes);
  return local.map((v, i) => origin[i] + (v - model.localCenter[i]) * model.unit);
}

export function toLocal(model, point, axes = frontal) {
  const origin = modelOrigin(model, axes);
  return point.map((v, i) => model.localCenter[i] + (v - origin[i]) / model.unit);
}

export function localView(model, view) {
  return cameraMap(view, toLocal(model, view.target, view), view.scale * model.unit, view.w, view.h);
}
