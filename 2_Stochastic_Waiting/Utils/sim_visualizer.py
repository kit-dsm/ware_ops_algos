from typing import Optional
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
import matplotlib.cm as cm
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button, RadioButtons


class SimulationVisualizer:
    def __init__(self, warehouse, log):
        """
        :param warehouse: Warehouse object containing a travelarea attribute with a NetworkX graph.
        :param log: List of snapshots (e.g., [{"time": datetime_obj, "pickers": [...]}, ...]).
        """
        self.warehouse = warehouse
        self.log = log
        self.interval = 1                     # Default interval: as fast as possible (1 ms)
        self.num_frames = len(log)
        self.current_frame = 0                # Current frame index
        self.is_playing = False               # Initially not in Play mode
        self._slider_updating = False         # Flag to distinguish internal slider updates from callback events
        self._slider_was_playing = False      # Flag to store if animation was playing before slider movement
        self.anim = None                      # No FuncAnimation object created yet

        # Retrieve the graph and positions from the warehouse's travelarea
        self.G = warehouse.graph
        self.pos = nx.get_node_attributes(self.G, "pos")

        # Create the figure and main axes.
        # Adjust subplots to move the main animation area higher (leaving room at the bottom for UI widgets).
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.12)
        nx.draw(self.G, self.pos, ax=self.ax, with_labels=True,
                node_size=300, font_size=4, font_color="w")

        # Initialize the dynamic artists based on the first snapshot
        first_snapshot = log[0]
        self.picker_markers = {
            picker.id: self.ax.plot([], [], "o", label=f"Picker {picker.id}",
                                      color=self.id_to_color(pid=picker.id), markersize=13)[0]
            for picker in first_snapshot["pickers"]
        }
        self.status_texts = {
            picker.id: self.ax.text(0, 0, "", fontsize=8, color="red")
            for picker in first_snapshot["pickers"]
        }
        self.event_text = self.ax.text(0.01, 0.95, "", transform=self.ax.transAxes, fontsize=10)
        self.time_text = self.ax.text(0.01, 0.91, "", transform=self.ax.transAxes, fontsize=10)

        self.picker_routes = {
            picker.id: self.ax.plot([], [], "-", lw=3, color=self.id_to_color(picker.id), alpha=0.6)[0]
            for picker in first_snapshot["pickers"]
        }

        # Create a slider for manual frame control.
        # Placed entirely in the lower area, which does not overlap the main animation.
        ax_slider = self.fig.add_axes([0.06, 0.08, 0.73, 0.03], facecolor='lightgoldenrodyellow')
        self.slider = Slider(ax_slider, 'Frame', 0, self.num_frames - 1, valinit=0, valstep=1)
        self.slider.on_changed(self.on_slider_change)

        # Connect to the button release event for the slider axis
        self.fig.canvas.mpl_connect('button_release_event', self.on_slider_release)

        # Create the Play/Pause button.
        ax_button = self.fig.add_axes([0.06, 0.03, 0.10, 0.04], facecolor='lightgoldenrodyellow')
        self.button = Button(ax_button, 'Play')
        self.button.on_clicked(self.toggle_animation)

        # Create the Previous Frame button (symbol "<")
        ax_prev = self.fig.add_axes([0.18, 0.03, 0.04, 0.04], facecolor='lightgoldenrodyellow')
        self.prev_button = Button(ax_prev, '<')
        self.prev_button.on_clicked(self.on_prev_button)

        # Create the Next Frame button (symbol ">")
        ax_next = self.fig.add_axes([0.23, 0.03, 0.04, 0.04], facecolor='lightgoldenrodyellow')
        self.next_button = Button(ax_next, '>')
        self.next_button.on_clicked(self.on_next_button)

        # Create a speed selection menu using RadioButtons.
        ax_speed = self.fig.add_axes([0.84, 0.01, 0.15, 0.10], facecolor='lightgoldenrodyellow')
        speed_options = ["as fast as possible", "10 Frames/s", "5 Frames/s", "2 Frames/s"]
        self.speed_dropdown = RadioButtons(ax_speed, speed_options)
        self.speed_dropdown.on_clicked(self.on_speed_change)
        self.speed_dropdown.set_active(0)  # Set the default selection to "as fast as possible"

        # Initial update call (displays the first frame)
        self.update(0)
        self.fig.canvas.draw()

    def update(self, frame):
        """
        Updates the current frame.
        Reads the snapshot from the log and updates:
          - the timestamp text,
          - the positions and status texts of the pickers,
          - the slider (if it is not already updating internally).
        If the last frame is reached, the animation is automatically paused,
        the button label is set to "Play", and the current frame resets to 0.
        """
        self.current_frame = frame
        snapshot = self.log[frame]
        event = snapshot["event"]
        frame_time = snapshot["time"]
        frame_pickers = snapshot["pickers"]

        # Update the event and timestamp display
        self.event_text.set_text(f"Event: {event}")
        self.time_text.set_text(f"Time: {frame_time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Update picker positions and their status displays
        position_picker_counter = {}
        for picker in frame_pickers:
            pid = picker.id
            x, y = picker.location
            self.picker_markers[pid].set_data([x], [y])
            position = (x, y)
            position_picker_counter[position] = position_picker_counter.get(position, 0) + 1
            y_offset = 0.25 * position_picker_counter[position]
            self.status_texts[pid].set_position((x + 0.06, y + y_offset))

            transported_batch = getattr(picker, "transported_batch", None)
            self.status_texts[pid].set_text(
                f"P{pid} Avail.: {picker.available}" +
                (f", Batch: {transported_batch.id}" if transported_batch is not None else "")
            )

            route = None
            if transported_batch is not None:
                routing_result = getattr(transported_batch, "routing_result", None)
                if routing_result is not None and "tour" in routing_result and routing_result["tour"]:
                    route = routing_result["tour"]

            if route and len(route) > 1:
                xs = [self.pos[n][0] for n in route]
                ys = [self.pos[n][1] for n in route]
                self.picker_routes[pid].set_data(xs, ys)
                self.picker_routes[pid].set_visible(True)
            else:
                self.picker_routes[pid].set_data([], [])
                self.picker_routes[pid].set_visible(False)

        # Update the slider if its value does not match the current frame
        if self.slider.val != frame:
            self._slider_updating = True
            self.slider.set_val(frame)
            self._slider_updating = False

        # Automatically pause the animation when the last frame is reached
        if frame == self.num_frames - 1:
            if self.anim is not None:
                self.anim.event_source.stop()
            self.button.label.set_text("Play")
            self.is_playing = False

        return (list(self.picker_markers.values()) + list(self.status_texts.values()) +
                list(self.picker_routes.values()) + [self.event_text, self.time_text])

    def on_slider_change(self, val):
        """
        Called when the slider value is changed.
        Pauses the animation temporarily if it is playing, and updates the frame.
        The animation will be resumed only after the slider is released.
        """
        if self._slider_updating:
            return

        # If the animation is currently playing, pause it and store the state
        if self.is_playing:
            if self.anim is not None:
                self.anim.event_source.stop()
            self._slider_was_playing = True
            self.is_playing = False

        # Update the frame based on the slider position
        frame = int(val)
        self.update(frame)
        self.fig.canvas.draw_idle()

    def on_slider_release(self, event):
        """
        Called when a mouse button is released.
        If the release event occurs in the slider's area and the animation was playing before,
        resume the animation.
        """
        if event.inaxes == self.slider.ax:
            if self._slider_was_playing:
                if self.current_frame < self.num_frames - 1:
                    self.toggle_animation(None)
                self._slider_was_playing = False

    def on_prev_button(self, event):
        """
        Callback for the Previous Frame button.
        If the simulation is playing, pause it, then jump one frame backward (if possible).
        """
        if self.is_playing:
            if self.anim is not None:
                self.anim.event_source.stop()
            self.button.label.set_text("Play")
            self.is_playing = False

        new_frame = max(self.current_frame - 1, 0)
        self.slider.set_val(new_frame)

    def on_next_button(self, event):
        """
        Callback for the Next Frame button.
        If the simulation is playing, pause it, then jump one frame forward (if possible).
        """
        if self.is_playing:
            if self.anim is not None:
                self.anim.event_source.stop()
            self.button.label.set_text("Play")
            self.is_playing = False

        new_frame = min(self.current_frame + 1, self.num_frames - 1)
        self.slider.set_val(new_frame)

    def on_speed_change(self, label):
        """
        Callback for the speed selection menu.
        Sets the animation interval according to the selected option.
        If the simulation is running, the animation is restarted with the new interval immediately.
        """
        speed_mapping = {
            "as fast as possible": 1,
            "10 Frames/s": 100,
            "5 Frames/s": 200,
            "2 Frames/s": 500
        }
        self.interval = speed_mapping[label]
        # If the simulation is running, restart the animation with the new speed
        if self.anim is not None and self.is_playing:
            # Stop the current animation
            self.anim.event_source.stop()
            # Re-create the animation starting from the current frame with the new interval
            self.anim = FuncAnimation(
                self.fig,
                self.update,
                frames=range(self.current_frame, self.num_frames),
                interval=self.interval,
                blit=False,
                repeat=False
            )
        self.fig.canvas.draw_idle()

    def toggle_animation(self, event):
        """
        Toggles the animation. When "Play" is pressed, a new FuncAnimation is created,
        starting from the current frame until the end of the log (or automatically pausing
        and resetting at the last frame). When "Pause" is pressed, the timer is stopped.
        """
        if self.is_playing:
            # Pause: Stop the animation
            if self.anim is not None and hasattr(self.anim, 'event_source'):
                self.anim.event_source.stop()
            self.button.label.set_text("Play")
            self.is_playing = False
        else:
            # Play: if the last frame is reached, reset to the beginning
            if self.current_frame == self.num_frames - 1:
                self.slider.set_val(0)
                self.current_frame = 0
            # Start the animation from the current frame until the end of the log
            self.anim = FuncAnimation(
                self.fig,
                self.update,
                frames=range(self.current_frame, self.num_frames),
                interval=self.interval,
                blit=False,
                repeat=False
            )
            self.button.label.set_text("Pause")
            self.is_playing = True
        self.fig.canvas.draw()

    def show(self, block:Optional[bool] = False):
        """
        Displays the visualization.
        """
        if block: plt.show(block=True)
        else: plt.show()

    def id_to_color(self, pid, max_ids =10):
        cmap = cm.get_cmap('tab10', max_ids)
        return cmap(pid % max_ids)