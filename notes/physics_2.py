"""
Interactive Lens Physics Simulator
Demonstrates optical behavior of thin lenses with ray tracing and image formation
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button
from matplotlib.patches import FancyBboxPatch, FancyArrow
import matplotlib.patches as mpatches

class LensSimulator:
    def __init__(self):
        # Default parameters
        self.focal_length = 10.0  # cm
        self.object_distance = 15.0  # cm
        self.object_height = 5.0  # cm
        self.lens_type = 'convex'  # 'convex' or 'concave'
        self.refractive_index = 1.5

        # Calculate derived properties
        self.update_calculations()

        # Setup the figure
        self.setup_figure()

    def update_calculations(self):
        """Calculate image properties using thin lens equation"""
        # Adjust focal length sign based on lens type
        f = self.focal_length if self.lens_type == 'convex' else -self.focal_length

        # Thin lens equation: 1/f = 1/do + 1/di
        # Solve for image distance: di = (do * f) / (do - f)
        if abs(self.object_distance - f) < 0.01:
            self.image_distance = float('inf')
            self.image_height = float('inf')
            self.magnification = float('inf')
        else:
            self.image_distance = (self.object_distance * f) / (self.object_distance - f)
            # Magnification: m = -di/do = hi/ho
            self.magnification = -self.image_distance / self.object_distance
            self.image_height = self.magnification * self.object_height

    def setup_figure(self):
        """Setup the matplotlib figure with controls"""
        self.fig = plt.figure(figsize=(20, 12))
        self.fig.suptitle('Interactive Lens Physics Simulator', fontsize=18, fontweight='bold')

        # Main plot area for ray diagram
        self.ax_main = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)

        # Info panel
        self.ax_info = plt.subplot2grid((3, 3), (0, 2), rowspan=1)
        self.ax_info.axis('off')

        # Lens diagram panel
        self.ax_lens = plt.subplot2grid((3, 3), (1, 2), rowspan=1)
        self.ax_lens.axis('off')

        # Control sliders
        slider_color = 'lightgoldenrodyellow'

        ax_focal = plt.subplot2grid((3, 3), (2, 0), colspan=2)
        ax_focal.set_position([0.15, 0.25, 0.5, 0.02])
        self.slider_focal = Slider(ax_focal, 'Focal Length (cm)', 5, 50, valinit=self.focal_length, color=slider_color)

        ax_obj_dist = plt.subplot2grid((3, 3), (2, 1), colspan=2)
        ax_obj_dist.set_position([0.15, 0.20, 0.5, 0.02])
        self.slider_obj_dist = Slider(ax_obj_dist, 'Object Distance (cm)', 1, 100, valinit=self.object_distance, color=slider_color)

        ax_obj_height = plt.subplot2grid((3, 3), (2, 2), colspan=2)
        ax_obj_height.set_position([0.15, 0.15, 0.5, 0.02])
        self.slider_obj_height = Slider(ax_obj_height, 'Object Height (cm)', 1, 20, valinit=self.object_height, color=slider_color)

        ax_refr_index = plt.subplot2grid((3, 3), (2, 2), colspan=2)
        ax_refr_index.set_position([0.15, 0.10, 0.5, 0.02])
        self.slider_refr = Slider(ax_refr_index, 'Refractive Index', 1.1, 2.5, valinit=self.refractive_index, color=slider_color)

        # Radio buttons for lens type
        ax_radio = plt.axes([0.75, 0.15, 0.15, 0.15])
        self.radio = RadioButtons(ax_radio, ('Convex\n(Собирающая Линза)', 'Concave\n(Выпуклая Линза)'), active=0)

        # Reset button
        ax_reset = plt.axes([0.75, 0.05, 0.15, 0.04])
        self.btn_reset = Button(ax_reset, 'Reset', color='lightcoral', hovercolor='coral')

        # Connect events
        self.slider_focal.on_changed(self.update)
        self.slider_obj_dist.on_changed(self.update)
        self.slider_obj_height.on_changed(self.update)
        self.slider_refr.on_changed(self.update)
        self.radio.on_clicked(self.update_lens_type)
        self.btn_reset.on_clicked(self.reset)

        # Initial plot
        self.plot()

    def update(self, val=None):
        """Update the simulation when parameters change"""
        self.focal_length = self.slider_focal.val
        self.object_distance = self.slider_obj_dist.val
        self.object_height = self.slider_obj_height.val
        self.refractive_index = self.slider_refr.val

        self.update_calculations()
        self.plot()

    def update_lens_type(self, label):
        """Update lens type from radio button"""
        if 'Convex' in label:
            self.lens_type = 'convex'
        else:
            self.lens_type = 'concave'
        self.update()

    def reset(self, event):
        """Reset to default values"""
        self.slider_focal.reset()
        self.slider_obj_dist.reset()
        self.slider_obj_height.reset()
        self.slider_refr.reset()
        self.radio.set_active(0)

    def draw_lens(self, ax, x_pos=0):
        """Draw the lens shape"""
        lens_height = 40
        lens_width = 3

        if self.lens_type == 'convex':
            # Convex lens (biconvex)
            theta = np.linspace(-np.pi/2, np.pi/2, 50)
            radius = lens_height / 2
            curve_depth = lens_width / 2

            # Left curve
            x_left = x_pos - curve_depth * np.cos(theta)
            y_left = radius * np.sin(theta)

            # Right curve
            x_right = x_pos + curve_depth * np.cos(theta)
            y_right = radius * np.sin(theta)

            ax.fill_betweenx(y_left, x_left, x_right, alpha=0.3, color='lightblue', edgecolor='blue', linewidth=2)

        else:
            # Concave lens (biconcave)
            theta = np.linspace(-np.pi/2, np.pi/2, 50)
            radius = lens_height / 2
            curve_depth = lens_width / 2

            # Left curve (inverted)
            x_left = x_pos + curve_depth * np.cos(theta)
            y_left = radius * np.sin(theta)

            # Right curve (inverted)
            x_right = x_pos - curve_depth * np.cos(theta)
            y_right = radius * np.sin(theta)

            ax.fill_betweenx(y_left, x_left, x_right, alpha=0.3, color='lightcoral', edgecolor='red', linewidth=2)

        # Draw principal axis
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)

    def draw_ray_diagram(self, ax):
        """Draw complete ray diagram with object, lens, and image"""
        ax.clear()

        # Set up coordinate system (lens at x=0)
        lens_x = 0
        object_x = -self.object_distance

        # Draw lens
        self.draw_lens(ax, lens_x)

        # Draw focal points
        f = self.focal_length if self.lens_type == 'convex' else -self.focal_length

        # Left focal point
        ax.plot([-abs(f)], [0], 'rx', markersize=12, markeredgewidth=3, label='Focal Point F')
        ax.text(-abs(f), -2, 'F', ha='center', va='top', fontsize=12, fontweight='bold')

        # Right focal point
        ax.plot([abs(f)], [0], 'rx', markersize=12, markeredgewidth=3)
        ax.text(abs(f), -2, "F'", ha='center', va='top', fontsize=12, fontweight='bold')

        # Draw object
        ax.arrow(object_x, 0, 0, self.object_height, head_width=1.5, head_length=0.7,
                fc='green', ec='green', linewidth=3, label='Object')
        ax.plot([object_x], [0], 'go', markersize=10)

        # Draw three principal rays
        ray_colors = ['red', 'blue', 'purple']
        ray_labels = ['Ray 1: Parallel → through F', 'Ray 2: Through center', 'Ray 3: Through F → parallel']

        # Ray 1: Parallel to axis, passes through focal point after refraction
        if self.lens_type == 'convex':
            ax.plot([object_x, lens_x], [self.object_height, self.object_height],
                   color=ray_colors[0], linewidth=2.5, linestyle='-', alpha=0.7)
            if abs(self.image_distance) < 200:
                ax.plot([lens_x, lens_x + abs(f)], [self.object_height, 0],
                       color=ray_colors[0], linewidth=2.5, linestyle='-', alpha=0.7)
        else:
            ax.plot([object_x, lens_x], [self.object_height, self.object_height],
                   color=ray_colors[0], linewidth=2.5, linestyle='-', alpha=0.7)
            # Diverging ray appears to come from focal point
            slope = self.object_height / abs(f)
            x_end = 50
            y_end = self.object_height + slope * x_end
            ax.plot([lens_x, x_end], [self.object_height, y_end],
                   color=ray_colors[0], linewidth=2.5, linestyle='-', alpha=0.7)
            # Virtual ray
            ax.plot([lens_x, -abs(f)], [self.object_height, 0],
                   color=ray_colors[0], linewidth=2.5, linestyle='--', alpha=0.5)

        # Ray 2: Through optical center (no deviation)
        if abs(self.image_distance) < 200:
            image_x = self.image_distance
            slope = self.object_height / self.object_distance
            y_at_lens = slope * (-lens_x + self.object_distance)
            ax.plot([object_x, 50], [self.object_height, self.object_height - slope * (50 + self.object_distance)],
                   color=ray_colors[1], linewidth=2.5, linestyle='-', alpha=0.7)

        # Ray 3: Through focal point, emerges parallel
        if self.lens_type == 'convex' and self.object_distance > abs(f):
            ax.plot([object_x, -abs(f)], [self.object_height, 0],
                   color=ray_colors[2], linewidth=2.5, linestyle='-', alpha=0.7)
            if abs(self.image_distance) < 200:
                ax.plot([-abs(f), 50], [0, 0],
                       color=ray_colors[2], linewidth=2.5, linestyle='-', alpha=0.7)

        # Draw image if it exists and is real
        if abs(self.image_distance) < 200 and not np.isinf(self.image_distance):
            image_x = self.image_distance

            if self.image_distance > 0:  # Real image
                ax.arrow(image_x, 0, 0, self.image_height, head_width=1.5, head_length=0.7,
                        fc='orange', ec='orange', linewidth=3, label='Real Image')
                ax.plot([image_x], [0], 'o', color='orange', markersize=10)
            else:  # Virtual image
                ax.arrow(image_x, 0, 0, self.image_height, head_width=1.5, head_length=0.7,
                        fc='cyan', ec='cyan', linewidth=3, linestyle='--', alpha=0.7, label='Virtual Image')
                ax.plot([image_x], [0], 'o', color='cyan', markersize=10)

        # Set axis limits
        max_dist = max(abs(self.object_distance), abs(f) * 3, 50)
        ax.set_xlim(-max_dist, max_dist)
        ax.set_ylim(-20, 25)

        ax.set_xlabel('Distance (cm)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Height (cm)', fontsize=13, fontweight='bold')
        ax.set_title('Ray Diagram', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)
        ax.set_aspect('equal', adjustable='box')

    def display_info(self, ax):
        """Display calculated information"""
        ax.clear()
        ax.axis('off')

        # Prepare information text
        info_text = []
        info_text.append("═══ LENS PROPERTIES ═══")
        info_text.append(f"Type: {self.lens_type.capitalize()}")
        info_text.append(f"Focal Length: {abs(self.focal_length):.2f} cm")
        info_text.append(f"Refractive Index: {self.refractive_index:.2f}")
        info_text.append("")
        info_text.append("═══ OBJECT ═══")
        info_text.append(f"Distance: {self.object_distance:.2f} cm")
        info_text.append(f"Height: {self.object_height:.2f} cm")
        info_text.append("")
        info_text.append("═══ IMAGE ═══")

        if np.isinf(self.image_distance):
            info_text.append("Distance: ∞ (at infinity)")
            info_text.append("Height: ∞")
            info_text.append("Type: No image formed")
            info_text.append("(Object at focal point)")
        elif abs(self.image_distance) > 1000:
            info_text.append("Distance: Very far")
            info_text.append("Image forms at infinity")
        else:
            info_text.append(f"Distance: {abs(self.image_distance):.2f} cm")
            info_text.append(f"Height: {abs(self.image_height):.2f} cm")

            if self.image_distance > 0:
                info_text.append("Type: REAL (inverted)")
                info_text.append("Side: Opposite to object")
            else:
                info_text.append("Type: VIRTUAL (upright)")
                info_text.append("Side: Same as object")

            info_text.append(f"Magnification: {self.magnification:.3f}×")

            if abs(self.magnification) > 1:
                info_text.append("Size: Magnified")
            elif abs(self.magnification) < 1:
                info_text.append("Size: Diminished")
            else:
                info_text.append("Size: Same size")

        info_text.append("")
        info_text.append("═══ EQUATIONS ═══")
        info_text.append("1/f = 1/do + 1/di")
        info_text.append("m = -di/do = hi/ho")

        # Display text
        y_pos = 0.95
        for line in info_text:
            if "═══" in line:
                ax.text(0.05, y_pos, line, fontsize=11, fontweight='bold',
                       family='monospace', transform=ax.transAxes)
            else:
                ax.text(0.05, y_pos, line, fontsize=10, family='monospace',
                       transform=ax.transAxes)
            y_pos -= 0.05

    def draw_lens_shape_diagram(self, ax):
        """Draw lens cross-section with annotations"""
        ax.clear()
        ax.axis('off')
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)

        if self.lens_type == 'convex':
            # Draw biconvex lens
            theta = np.linspace(-np.pi/2, np.pi/2, 50)
            x_left = -0.3 * np.cos(theta)
            y = np.sin(theta)
            x_right = 0.3 * np.cos(theta)

            ax.fill_betweenx(y, x_left, x_right, alpha=0.4, color='lightblue',
                            edgecolor='blue', linewidth=2)
            ax.text(0, -1.5, 'Convex Lens\n(Собирающая Линза)', ha='center', fontsize=11,
                   fontweight='bold', color='blue')

        else:
            # Draw biconcave lens
            theta = np.linspace(-np.pi/2, np.pi/2, 50)
            x_left = 0.3 * np.cos(theta)
            y = np.sin(theta)
            x_right = -0.3 * np.cos(theta)

            ax.fill_betweenx(y, x_left, x_right, alpha=0.4, color='lightcoral',
                            edgecolor='red', linewidth=2)
            ax.text(0, -1.5, 'Concave Lens\n(Выпуклая Линза)', ha='center', fontsize=11,
                   fontweight='bold', color='red')

        ax.set_title('Lens Cross-Section', fontsize=12, fontweight='bold')

    def plot(self):
        """Main plotting function"""
        self.draw_ray_diagram(self.ax_main)
        self.display_info(self.ax_info)
        self.draw_lens_shape_diagram(self.ax_lens)
        self.fig.canvas.draw_idle()

def main():
    """Main function to run the simulator"""
    print("=" * 60)
    print("Interactive Lens Physics Simulator")
    print("=" * 60)
    print("\nControls:")
    print("- Use sliders to adjust lens and object properties")
    print("- Select lens type (Convex or Concave)")
    print("- Watch how the image changes in real-time")
    print("- Click 'Reset' to restore default values")
    print("\nKey Concepts:")
    print("- Convex lenses converge light rays")
    print("- Concave lenses diverge light rays")
    print("- Real images: opposite side, inverted")
    print("- Virtual images: same side, upright")
    print("\nClose the window to exit.")
    print("=" * 60)

    simulator = LensSimulator()
    plt.show()

if __name__ == "__main__":
    main()