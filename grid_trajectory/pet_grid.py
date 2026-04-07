from collections import defaultdict

class TrajectoryLogger:
    def __init__(self, fps):
        self.fps = fps
        # track_id:int -> list of (frame_idx:int, cell_id:any, world_x:any, world_y:any)
        self.tracks = defaultdict(list)

    def log(self, track_id, frame_idx, cell_id, world_x=None, world_y=None):
        tid = int(track_id)
        fi = int(frame_idx)
        self.tracks[tid].append((fi, cell_id, world_x, world_y))

    def build_intervals(self):
        intervals = []
        for obj_id, samples in self.tracks.items():
            samples.sort(key=lambda x: x[0])

            prev_cell = None
            start_frame = None
            world_samples = []

            for frame_idx, cell_id, wx, wy in samples:
                fi = frame_idx

                if prev_cell is None:
                    prev_cell = cell_id
                    start_frame = fi
                    if wx is not None and wy is not None:
                        world_samples.append((fi / self.fps, float(wx), float(wy)))
                    continue

                if cell_id == prev_cell:
                    if wx is not None and wy is not None:
                        world_samples.append((fi / self.fps, float(wx), float(wy)))
                else:
                    end_frame = fi - 1
                    intervals.append(dict(
                        obj_id=obj_id,
                        cell_id=prev_cell,
                        t_enter=start_frame / self.fps,
                        t_exit=end_frame / self.fps,
                        world_samples=world_samples,
                    ))
                    prev_cell = cell_id
                    start_frame = fi
                    world_samples = []
                    if wx is not None and wy is not None:
                        world_samples.append((fi / self.fps, float(wx), float(wy)))

            if prev_cell is not None and start_frame is not None:
                end_frame = samples[-1][0]
                intervals.append(dict(
                    obj_id=obj_id,
                    cell_id=prev_cell,
                    t_enter=start_frame / self.fps,
                    t_exit=end_frame / self.fps,
                    world_samples=world_samples,
                ))

        return intervals


def compute_pet(intervals, pet_threshold=2.0):
    pet_events = []
    by_cell = {}

    for iv in intervals:
        by_cell.setdefault(iv["cell_id"], []).append(iv)

    for cell_id, cell_intervals in by_cell.items():
        n = len(cell_intervals)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                A = cell_intervals[i]
                B = cell_intervals[j]

                if A["t_exit"] <= B["t_enter"]:
                    pet = B["t_enter"] - A["t_exit"]
                    if 0 < pet <= pet_threshold:
                        pet_events.append(dict(
                            obj_i=A["obj_id"],
                            obj_j=B["obj_id"],
                            cell_id=cell_id,
                            t_exit_i=A["t_exit"],
                            t_entry_j=B["t_enter"],
                            PET=pet,
                            world_traj_i=A.get("world_samples", []),
                            world_traj_j=B.get("world_samples", []),
                        ))

    return pet_events
