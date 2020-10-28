import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib import cm
from tqdm import tqdm

from typing import Dict, Sequence


def plot_video_actiavation_muscle(
    activation_list: dict,
    torque_list: dict,
    video_name="video.mp4",
    margin=0.2,
    fps=15,
    step=100,
):
    import matplotlib.animation as manimation

    time = np.array(activation_list["time"])
    if "activation_signal" in activation_list:
        activation = np.array(activation_list["activation_signal"])
        first_activation = None
        second_activation = None
    else:
        activation = None
        first_activation = np.array(activation_list["first_activation_signal"])
        second_activation = np.array(activation_list["second_activation_signal"])

    torque = np.array(torque_list["torque"])
    if "torque_mag" in torque_list:
        torque_mag = np.array(torque_list["torque_mag"])
        first_torque_mag = None
        second_torque_mag = None
    else:
        torque_mag = None
        first_torque_mag = np.array(torque_list["first_torque_mag"])
        second_torque_mag = np.array(torque_list["second_torque_mag"])
    element_position = np.array(torque_list["element_position"])

    print("plot activation visualization video")
    FFMpegWriter = manimation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    fig = plt.figure()
    plt.subplot(311)
    plt.axis("equal")
    with writer.saving(fig, video_name, 100):
        for time in tqdm(range(1, time.shape[0], int(step))):
            # ax1 = plt.subplot(2, 2, 1)
            # ax2 = plt.subplot(222, frameon=False)
            # x = activation[time][2]
            torq = torque[time][1]
            pos = element_position[time]
            fig.clf()
            plt.subplot(3, 1, 1)
            if activation is not None:
                plt.bar(activation[time])
            else:
                plt.bar(np.arange(len(first_activation[time])),first_activation[time])
                plt.bar(np.arange(len(second_activation[time])),second_activation[time])
                # plt.plot(second_activation[time])
            plt.ylim([-1 - margin, 1 + margin])
            plt.subplot(3, 1, 2)

            if torque_mag is not None:
                plt.plot(pos, torque_mag[time], "-")
            else:
                plt.plot(pos, first_torque_mag[time], pos, second_torque_mag[time])
            plt.subplot(3, 1, 3)
            plt.plot(pos, torq, "-")
            # plt.xlim([0 - margin, 2.5 + margin])

            writer.grab_frame()


def plot_video(
    plot_params: dict,
    video_name="video.mp4",
    margin=0.2,
    fps=15,
    step_skip=100,
    *args,
    **kwargs,
):  # (time step, x/y/z, node)
    import matplotlib.animation as manimation

    positions_over_time = np.array(plot_params["position"])

    print("plot video")
    FFMpegWriter = manimation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata, bitrate=-1)
    fig = plt.figure(figsize=(10, 8), frameon=True, dpi=150)
    plt.gca().set_aspect("equal", adjustable="box")
    time = np.array(plot_params["time"])
    with writer.saving(fig, video_name, 100):
        for time in range(1, time.shape[0], int(step_skip)):
            x = positions_over_time[time][2]
            y = positions_over_time[time][1]
            fig.clf()
            if kwargs.__contains__("target"):
                plt.plot(kwargs["target"][2], kwargs["target"][1], "*", markersize=12)
            plt.plot(x, y, "o")
            plt.gca().set_aspect("equal", adjustable="box")
            plt.xlim([-1.0 - margin, 1.0 + margin])
            plt.ylim([-1.0 - margin, 1.0 + margin])
            writer.grab_frame()


def plot_force(
    end_force_recorder: dict,
    file_name="force.png",
    margin=0.2,
    fps=15,
    SAVE_FIGURE=True,
):

    time = np.array(end_force_recorder["time"])
    force = np.array(end_force_recorder["action"])
    # print(force.shape)

    print("plot force")
    fig, axs = plt.subplots(3, 1, constrained_layout=False)
    axs[0].plot(time, force[:, 0], "-")
    axs[0].set_title("force value - x/y/z")

    axs[1].plot(time, force[:, 1], "-")
    axs[1].set_ylabel("force value")

    axs[2].plot(time, force[:, 2], "-")
    axs[2].set_xlabel("time [s]")

    # plt.show()
    if SAVE_FIGURE:
        fig.savefig(file_name)

    plt.close()


def plot_factor(
    end_force_recorder: dict,
    file_name="force_factor.png",
    margin=0.2,
    fps=15,
    SAVE_FIGURE=True,
):

    time = np.array(end_force_recorder["time"])
    factor = np.array(end_force_recorder["action_coef"])

    print("plot action factor")
    print(factor.shape)
    fig, axs = plt.subplots(3, 1, constrained_layout=False)

    axs[0].plot(time, factor[:, 0], "-")
    axs[0].set_title("force factor value - x/y/z")

    #################### currently modifying factor plotting function ####################
    axs[1].plot(time, factor[:, 1], "-")
    axs[1].set_ylabel("force factor value")

    axs[2].plot(time, factor[:, 2], "-")
    axs[2].set_xlabel("time [s]")

    # plt.show()
    if SAVE_FIGURE:
        fig.savefig(file_name)

    plt.close()


def plot_torque(
    torque_recorder: dict, video_name="torque.mp4", margin=0.2, fps=15, SAVE_FIGURE=True
):

    import matplotlib.animation as manimation

    time = np.array(torque_recorder["time"])
    torque = np.array(torque_recorder["torque"])
    torque_mag = np.array(torque_recorder["torque_mag"])
    element_position = np.array(torque_recorder["element_position"])

    print("plot torque")
    FFMpegWriter = manimation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    fig = plt.figure()
    plt.subplot(321)
    plt.axis("equal")
    plt.ylim([-1 - margin, 1 + margin])
    with writer.saving(fig, video_name, 100):
        for time in range(1, time.shape[0], int(step)):
            torq = torque[time][1]
            pos = element_position[time]
            fig.clf()
            plt.subplot(3, 1, 1)
            plt.plot(pos, torque[time][0], "-")
            plt.subplot(3, 1, 2)
            plt.plot(pos, torque_mag[time], "-")
            plt.subplot(3, 1, 3)
            plt.plot(pos, torque[time][1], "-")
            plt.subplot(3, 2, 1)
            plt.plot(pos, torq, "-")
            plt.subplot(3, 2, 2)
            plt.plot(pos, torque[time][2], "-")
            writer.grab_frame()


def plot_signal(sensor_recorder: dict, file_name, SAVE_FIGURE=True):

    time = np.array(sensor_recorder["time"])
    sensor_value = np.array(sensor_recorder["signal"])
    np.save("state", sensor_value)
    # print(sensor_value.shape)

    fig, axs = plt.subplots(3, 1, constrained_layout=False)
    axs[0].plot(time, sensor_value[:, 0], "-")
    axs[0].set_title("sensor value - x/y/z")

    axs[1].plot(time, sensor_value[:, 1], "-")
    axs[1].set_ylabel("sensor value")

    axs[2].plot(time, sensor_value[:, 2], "-")
    axs[2].set_xlabel("time [s]")

    # plt.show()
    if SAVE_FIGURE:
        fig.savefig(file_name)

    plt.close()


def plot_video3d(
    plot_params: dict,
    step_skip,
    video_name="video.mp4",
    margin=0.2,
    fps=20,
    *args,
    **kwargs,
):
    import matplotlib.animation as manimation
    from mpl_toolkits import mplot3d

    time = plot_params["time"]
    positions_over_time = np.array(plot_params["position"])

    print("plot video3d")
    FFMpegWriter = manimation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata, bitrate=-1)
    fig = plt.figure(figsize=(10, 8), frameon=True, dpi=150)
    with writer.saving(fig, video_name, step_skip):
        for time in range(1, len(time)):
            fig.clf()
            ax = plt.axes(projection="3d")  # fig.add_subplot(111)
            ax.grid(b=True, which="minor", color="k", linestyle="--")
            ax.grid(b=True, which="major", color="k", linestyle="-")
            if kwargs.__contains__("target"):
                ax.scatter(
                    kwargs["target"][2], kwargs["target"][0], kwargs["target"][1], "*"
                )
            ax.scatter(
                positions_over_time[time, 2],
                positions_over_time[time, 0],
                positions_over_time[time, 1],
                s=np.pi * 0.5 ** 2 * 10,
            )
            ax.set_xlim(-1.0 - margin, 1.0 + margin)
            ax.set_ylim(-1.0 - margin, 1.0 + margin)
            ax.set_zlim(-1.0 - margin, 1.0 + margin)
            # ax.set_xlabel("z positon")
            # ax.set_ylabel("x position")
            # ax.set_zlabel("y position")
            writer.grab_frame()


def plot_video_with_sphere(
    rods_history: Sequence[Dict],
    sphere_history: Sequence[Dict],
    video_name="video.mp4",
    fps=60,
    step=1,
    vis2D=True,
    **kwargs,
):
    plt.rcParams.update({"font.size": 22})

    # 2d case <always 2d case for now>
    import matplotlib.animation as animation
    from matplotlib.patches import Circle
    from mpl_toolkits.mplot3d import proj3d, Axes3D

    # simulation time
    sim_time = np.array(sphere_history[0]["time"])

    # Sphere
    n_visualized_spheres = len(sphere_history)  # should be one for now
    # Sphere radius
    sphere_radii = [x["radius"] for x in sphere_history]
    # Sphere info
    sphere_history_unpacker = lambda sph_idx, t_idx: (
        sphere_history[sph_idx]["position"][t_idx],
        sphere_history[sph_idx]["radius"][t_idx],
    )
    # color mapping
    sphere_cmap = cm.get_cmap("Spectral", n_visualized_spheres)

    # Rod
    n_visualized_rods = len(rods_history)  # should be one for now
    # Rod info
    rod_history_unpacker = lambda rod_idx, t_idx: (
        rods_history[rod_idx]["position"][t_idx],
        rods_history[rod_idx]["radius"][t_idx],
    )
    # Rod center of mass
    com_history_unpacker = lambda rod_idx, t_idx: rods_history[rod_idx]["com"][time_idx]

    # video pre-processing
    print("plot scene visualization video")
    FFMpegWriter = animation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    dpi = kwargs.get("dpi", 100)

    xlim = kwargs.get("x_limits", (-1.0, 1.0))
    ylim = kwargs.get("y_limits", (-1.0, 1.0))
    zlim = kwargs.get("z_limits", (-0.0, 1.0))

    difference = lambda x: x[1] - x[0]
    max_axis_length = max(difference(xlim), difference(ylim))
    # The scaling factor from physical space to matplotlib space
    scaling_factor = (2 * 0.1) / max_axis_length  # Octopus head dimension
    scaling_factor *= 2.6e3  # Along one-axis

    fig = plt.figure(2, figsize=(10, 8), frameon=True, dpi=dpi)
    # ax = fig.add_subplot(111)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)

    sphere_history_directors = lambda sph_idx, t_idx: (
        sphere_history[sph_idx]["directors"][t_idx],
    )
    rod_history_directors = lambda sph_idx, t_idx: (
        rods_history[sph_idx]["directors"][t_idx],
    )
    rod_directors1 = [None for _ in range(n_visualized_rods)]
    rod_directors2 = [None for _ in range(n_visualized_rods)]
    rod_directors3 = [None for _ in range(n_visualized_rods)]
    sphere_directors1 = [None for _ in range(n_visualized_spheres)]
    sphere_directors2 = [None for _ in range(n_visualized_spheres)]
    sphere_directors3 = [None for _ in range(n_visualized_spheres)]

    time_idx = 0
    rod_lines = [None for _ in range(n_visualized_rods)]
    rod_com_lines = [None for _ in range(n_visualized_rods)]
    rod_scatters = [None for _ in range(n_visualized_rods)]

    for rod_idx in range(n_visualized_rods):
        inst_position, inst_radius = rod_history_unpacker(rod_idx, time_idx)
        inst_position = 0.5 * (inst_position[..., 1:] + inst_position[..., :-1])
        # rod_lines[rod_idx] = ax.plot(inst_position[0], inst_position[1],inst_position[2], "r", lw=0.5)[0]
        inst_com = com_history_unpacker(rod_idx, time_idx)
        # rod_com_lines[rod_idx] = ax.plot(inst_com[0], inst_com[1],inst_com[2], "k--", lw=2.0)[0]

        rod_scatters[rod_idx] = ax.scatter(
            inst_position[2],
            inst_position[0],
            inst_position[1],
            s=np.pi * (scaling_factor * inst_radius) ** 2,
        )

        rod_dir = rod_history_directors(rod_idx, time_idx) 
        scale = 0.2
        dir1 = rod_dir[0][...,-1][0] * scale
        dir2 = rod_dir[0][...,-1][1] * scale
        dir3 = rod_dir[0][...,-1][2] * scale
        tip_position = inst_position[..., -1]
        rod_directors1[rod_idx] = ax.plot(
            np.array([tip_position[2], tip_position[2] + dir1[2]]).flatten(),
            np.array([tip_position[0], tip_position[0] + dir1[0]]).flatten(),
            np.array([tip_position[1], tip_position[1] + dir1[1]]).flatten(),
            "r-", lw=2.0)[0]
        rod_directors2[rod_idx] = ax.plot(
            np.array([tip_position[2], tip_position[2] + dir2[2]]).flatten(),
            np.array([tip_position[0], tip_position[0] + dir2[0]]).flatten(),
            np.array([tip_position[1], tip_position[1] + dir2[1]]).flatten(),
            "g-", lw=2.0)[0]
        rod_directors3[rod_idx] = ax.plot(
            np.array([tip_position[2], tip_position[2] + dir3[2]]).flatten(),
            np.array([tip_position[0], tip_position[0] + dir3[0]]).flatten(),
            np.array([tip_position[1], tip_position[1] + dir3[1]]).flatten(),
            "b-", lw=2.0)[0]


    sphere_artists = [None for _ in range(n_visualized_spheres)]
    for sphere_idx in range(n_visualized_spheres):
        sphere_position, sphere_radius = sphere_history_unpacker(sphere_idx, time_idx)
        sphere_artists[sphere_idx] = ax.scatter(
            sphere_position[2],
            sphere_position[0],
            sphere_position[1],
            s=np.pi * (scaling_factor * inst_radius) ** 2,
        )
        # sphere_radius,
        # color=sphere_cmap(sphere_idx),)
        ax.add_artist(sphere_artists[sphere_idx])

        sphere_dir = sphere_history_directors(sphere_idx, time_idx) 
        scale = 0.2
        dir1 = sphere_dir[0][0] * scale
        dir2 = sphere_dir[0][1] * scale
        dir3 = sphere_dir[0][2] * scale
        sphere_directors1[sphere_idx] = ax.plot(
            np.array([sphere_position[0], sphere_position[0] + dir1[0]]).flatten(),
            np.array([sphere_position[1], sphere_position[1] + dir1[1]]).flatten(),
            np.array([sphere_position[2], sphere_position[2] + dir1[2]]).flatten(),
            "r-", lw=2.0)[0]
        sphere_directors2[sphere_idx] = ax.plot(
            np.array([sphere_position[0], sphere_position[0] + dir2[0]]).flatten(),
            np.array([sphere_position[1], sphere_position[1] + dir2[1]]).flatten(),
            np.array([sphere_position[2], sphere_position[2] + dir2[2]]).flatten(),
            "g-", lw=2.0)[0]
        sphere_directors3[sphere_idx] = ax.plot(
            np.array([sphere_position[0], sphere_position[0] + dir3[0]]).flatten(),
            np.array([sphere_position[1], sphere_position[1] + dir3[1]]).flatten(),
            np.array([sphere_position[2], sphere_position[2] + dir3[2]]).flatten(),
            "b-", lw=2.0)[0]

    # ax.set_aspect("equal")
    video_name = "2D_" + video_name

    with writer.saving(fig, video_name, dpi):
        with plt.style.context("seaborn-whitegrid"):
            for time_idx in tqdm(range(0, sim_time.shape[0], int(step))):

                for rod_idx in range(n_visualized_rods):
                    inst_position, inst_radius = rod_history_unpacker(rod_idx, time_idx)
                    inst_position = 0.5 * (
                        inst_position[..., 1:] + inst_position[..., :-1]
                    )

                    # rod_lines[rod_idx].set_xdata(inst_position[0])
                    # rod_lines[rod_idx].set_ydata(inst_position[1])
                    # rod_lines[rod_idx].set_zdata(inst_position[2])

                    com = com_history_unpacker(rod_idx, time_idx)
                    # rod_com_lines[rod_idx].set_xdata(com[0])
                    # rod_com_lines[rod_idx].set_ydata(com[1])
                    # rod_com_lines[rod_idx].set_zdata(com[2])

                    # rod_scatters[rod_idx].set_offsets(inst_position[:3].T)
                    rod_scatters[rod_idx]._offsets3d = (
                        inst_position[2],
                        inst_position[0],
                        inst_position[1],
                    )

                    rod_scatters[rod_idx].set_sizes(
                        np.pi * (scaling_factor * inst_radius) ** 2 * 0.1
                    )

                    rod_dir = rod_history_directors(sphere_idx, time_idx) 
                    dir1 = rod_dir[0][...,-1][0] * scale
                    dir2 = rod_dir[0][...,-1][1] * scale
                    dir3 = rod_dir[0][...,-1][2] * scale
                    tip_position = inst_position[..., -1]
                    rod_directors1[rod_idx].set_xdata(np.array([tip_position[2], tip_position[2] + dir1[2]]).flatten())
                    rod_directors1[rod_idx].set_ydata(np.array([tip_position[0], tip_position[0] + dir1[0]]).flatten())
                    rod_directors1[rod_idx].set_3d_properties(np.array([tip_position[1], tip_position[1] + dir1[1]]).flatten())

                    rod_directors2[rod_idx].set_xdata(np.array([tip_position[2], tip_position[2] + dir2[2]]).flatten())
                    rod_directors2[rod_idx].set_ydata(np.array([tip_position[0], tip_position[0] + dir2[0]]).flatten())
                    rod_directors2[rod_idx].set_3d_properties(np.array([tip_position[1], tip_position[1] + dir2[1]]).flatten())

                    rod_directors3[rod_idx].set_xdata(np.array([tip_position[2], tip_position[2] + dir3[2]]).flatten())
                    rod_directors3[rod_idx].set_ydata(np.array([tip_position[0], tip_position[0] + dir3[0]]).flatten())
                    rod_directors3[rod_idx].set_3d_properties(np.array([tip_position[1], tip_position[1] + dir3[1]]).flatten())



                for sphere_idx in range(n_visualized_spheres):
                    sphere_position, _ = sphere_history_unpacker(sphere_idx, time_idx)
                    # sphere_artists[sphere_idx].center = (
                    #     sphere_position[0],
                    #     sphere_position[1],
                    #     sphere_position[2],
                    # )
                    # sphere_artists[sphere_idx].set_offsets(sphere_position[:3].T)
                    sphere_artists[sphere_idx]._offsets3d = (
                        sphere_position[2],
                        sphere_position[0],
                        sphere_position[1],
                    )

                    sphere_dir = sphere_history_directors(sphere_idx, time_idx) 
                    dir1 = sphere_dir[0][0] * scale
                    dir2 = sphere_dir[0][1] * scale
                    dir3 = sphere_dir[0][2] * scale

                    sphere_directors1[sphere_idx].set_xdata(np.array([sphere_position[2], sphere_position[2] + dir1[2]]).flatten())
                    sphere_directors1[sphere_idx].set_ydata(np.array([sphere_position[0], sphere_position[0] + dir1[0]]).flatten())
                    sphere_directors1[sphere_idx].set_3d_properties(np.array([sphere_position[1], sphere_position[1] + dir1[1]]).flatten())

                    sphere_directors2[sphere_idx].set_xdata(np.array([sphere_position[2], sphere_position[2] + dir2[2]]).flatten())
                    sphere_directors2[sphere_idx].set_ydata(np.array([sphere_position[0], sphere_position[0] + dir2[0]]).flatten())
                    sphere_directors2[sphere_idx].set_3d_properties(np.array([sphere_position[1], sphere_position[1] + dir2[1]]).flatten())       

                    sphere_directors3[sphere_idx].set_xdata(np.array([sphere_position[2], sphere_position[2] + dir3[2]]).flatten())
                    sphere_directors3[sphere_idx].set_ydata(np.array([sphere_position[0], sphere_position[0] + dir3[0]]).flatten())
                    sphere_directors3[sphere_idx].set_3d_properties(np.array([sphere_position[1], sphere_position[1] + dir3[1]]).flatten())


                writer.grab_frame()

    # Be a good boy and close figures
    # https://stackoverflow.com/a/37451036
    # plt.close(fig) alone does not suffice
    # See https://github.com/matplotlib/matplotlib/issues/8560/
    plt.close(plt.gcf())


def plot_video_with_sphere_2D(
    rods_history: Sequence[Dict],
    sphere_history: Sequence[Dict],
    video_name="video_2D.mp4",
    fps=60,
    step=1,
    vis2D=True,
    **kwargs,
):
    plt.rcParams.update({"font.size": 22})

    # 2d case <always 2d case for now>
    import matplotlib.animation as animation
    from matplotlib.patches import Circle

    # simulation time
    sim_time = np.array(sphere_history[0]["time"])

    # Sphere
    n_visualized_spheres = len(sphere_history)  # should be one for now
    # Sphere radius
    sphere_radii = [x["radius"] for x in sphere_history]
    # Sphere info
    sphere_history_unpacker = lambda sph_idx, t_idx: (
        sphere_history[sph_idx]["position"][t_idx],
        sphere_history[sph_idx]["radius"][t_idx],
    )

    # color mapping
    sphere_cmap = cm.get_cmap("Spectral", n_visualized_spheres)

    # Rod
    n_visualized_rods = len(rods_history)  # should be one for now
    # Rod info
    rod_history_unpacker = lambda rod_idx, t_idx: (
        rods_history[rod_idx]["position"][t_idx],
        rods_history[rod_idx]["radius"][t_idx],
    )

    # Rod center of mass
    com_history_unpacker = lambda rod_idx, t_idx: rods_history[rod_idx]["com"][time_idx]

    # video pre-processing
    print("plot scene visualization video")
    FFMpegWriter = animation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    dpi = kwargs.get("dpi", 100)

    xlim = kwargs.get("x_limits", (-1.1, 1.1))
    ylim = kwargs.get("y_limits", (-1.1, 1.1))
    zlim = kwargs.get("z_limits", (-0.05, 1.0))

    difference = lambda x: x[1] - x[0]
    max_axis_length = max(difference(xlim), difference(ylim))
    # The scaling factor from physical space to matplotlib space
    scaling_factor = (2 * 0.1) / max_axis_length  # Octopus head dimension
    scaling_factor *= 2.6e3  # Along one-axis

    fig = plt.figure(2, figsize=(10, 8), frameon=True, dpi=dpi)
    ax = fig.add_subplot(111)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    sphere_history_directors = lambda sph_idx, t_idx: (
        sphere_history[sph_idx]["directors"][t_idx],
    )
    rod_history_directors = lambda sph_idx, t_idx: (
        rods_history[sph_idx]["directors"][t_idx],
    )
    rod_directors1 = [None for _ in range(n_visualized_rods)]
    rod_directors2 = [None for _ in range(n_visualized_rods)]
    rod_directors3 = [None for _ in range(n_visualized_rods)]
    sphere_directors1 = [None for _ in range(n_visualized_spheres)]
    sphere_directors2 = [None for _ in range(n_visualized_spheres)]
    sphere_directors3 = [None for _ in range(n_visualized_spheres)]

    time_idx = 0
    rod_lines = [None for _ in range(n_visualized_rods)]
    rod_com_lines = [None for _ in range(n_visualized_rods)]
    rod_scatters = [None for _ in range(n_visualized_rods)]
    for rod_idx in range(n_visualized_rods):
        inst_position, inst_radius = rod_history_unpacker(rod_idx, time_idx)
        inst_position = 0.5 * (inst_position[..., 1:] + inst_position[..., :-1])
        rod_lines[rod_idx] = ax.plot(inst_position[0], inst_position[1], "r", lw=0.5)[0]
        inst_com = com_history_unpacker(rod_idx, time_idx)
        rod_com_lines[rod_idx] = ax.plot(inst_com[0], inst_com[1], "k--", lw=2.0)[0]

        rod_scatters[rod_idx] = ax.scatter(
            inst_position[0],
            inst_position[1],
            s=np.pi * (scaling_factor * inst_radius) ** 2,
        )

        rod_dir = rod_history_directors(rod_idx, time_idx) 
        scale = 0.2
        dir1 = rod_dir[0][...,-1][0] * scale
        dir2 = rod_dir[0][...,-1][1] * scale
        dir3 = rod_dir[0][...,-1][2] * scale
        tip_position = inst_position[..., -1]
        rod_directors1[rod_idx] = ax.plot(
            [tip_position[0], tip_position[0] + dir1[0]],
            [tip_position[1], tip_position[1] + dir1[1]],"r-", lw=2.0)[0]
        rod_directors2[rod_idx] = ax.plot(
            [tip_position[0], tip_position[0] + dir2[0]],
            [tip_position[1], tip_position[1] + dir2[1]],"g-", lw=2.0)[0]
        rod_directors3[rod_idx] = ax.plot(
            [tip_position[0], tip_position[0] + dir3[0]],
            [tip_position[1], tip_position[1] + dir3[1]],"b-", lw=2.0)[0]
        

    sphere_artists = [None for _ in range(n_visualized_spheres)]
    for sphere_idx in range(n_visualized_spheres):
        sphere_position, sphere_radius = sphere_history_unpacker(sphere_idx, time_idx)
        sphere_artists[sphere_idx] = Circle(
            (sphere_position[0], sphere_position[1]),
            sphere_radius,
            color=sphere_cmap(sphere_idx),
        )
        ax.add_artist(sphere_artists[sphere_idx])
        sphere_dir = sphere_history_directors(sphere_idx, time_idx) 
        scale = 0.2
        dir1 = sphere_dir[0][0] * scale
        dir2 = sphere_dir[0][1] * scale
        dir3 = sphere_dir[0][2] * scale
        sphere_directors1[sphere_idx] = ax.plot(
            [sphere_position[0], sphere_position[0] + dir1[0]],
            [sphere_position[1], sphere_position[1] + dir1[1]],"r-", lw=2.0)[0]
        sphere_directors2[sphere_idx] = ax.plot(
            [sphere_position[0], sphere_position[0] + dir2[0]],
            [sphere_position[1], sphere_position[1] + dir2[1]],"g-", lw=2.0)[0]
        sphere_directors3[sphere_idx] = ax.plot(
            [sphere_position[0], sphere_position[0] + dir3[0]],
            [sphere_position[1], sphere_position[1] + dir3[1]],"b-", lw=2.0)[0]

    ax.set_aspect("equal")
    video_name = "2D_" + video_name

    with writer.saving(fig, video_name, dpi):
        with plt.style.context("seaborn-whitegrid"):
            for time_idx in tqdm(range(0, sim_time.shape[0], int(step))):

                for rod_idx in range(n_visualized_rods):
                    inst_position, inst_radius = rod_history_unpacker(rod_idx, time_idx)
                    inst_position = 0.5 * (
                        inst_position[..., 1:] + inst_position[..., :-1]
                    )

                    rod_lines[rod_idx].set_xdata(inst_position[0])
                    rod_lines[rod_idx].set_ydata(inst_position[1])

                    com = com_history_unpacker(rod_idx, time_idx)
                    rod_com_lines[rod_idx].set_xdata(com[0])
                    rod_com_lines[rod_idx].set_ydata(com[1])

                    rod_scatters[rod_idx].set_offsets(inst_position[:2].T)
                    rod_scatters[rod_idx].set_sizes(
                        np.pi * (scaling_factor * inst_radius) ** 2
                    )

                    rod_dir = rod_history_directors(sphere_idx, time_idx) 
                    dir1 = rod_dir[0][...,-1][0] * scale
                    dir2 = rod_dir[0][...,-1][1] * scale
                    dir3 = rod_dir[0][...,-1][2] * scale
                    tip_position = inst_position[..., -1]
                    rod_directors1[rod_idx].set_xdata([tip_position[0], tip_position[0] + dir1[0]])
                    rod_directors1[rod_idx].set_ydata([tip_position[1], tip_position[1] + dir1[1]])

                    rod_directors2[rod_idx].set_xdata([tip_position[0], tip_position[0] + dir2[0]])
                    rod_directors2[rod_idx].set_ydata([tip_position[1], tip_position[1] + dir2[1]])        

                    rod_directors3[rod_idx].set_xdata([tip_position[0], tip_position[0] + dir3[0]])
                    rod_directors3[rod_idx].set_ydata([tip_position[1], tip_position[1] + dir3[1]])

                for sphere_idx in range(n_visualized_spheres):
                    sphere_position, _ = sphere_history_unpacker(sphere_idx, time_idx)
                    sphere_artists[sphere_idx].center = (
                        sphere_position[0],
                        sphere_position[1],
                    )

                    sphere_dir = sphere_history_directors(sphere_idx, time_idx) 
                    dir1 = sphere_dir[0][0] * scale
                    dir2 = sphere_dir[0][1] * scale
                    dir3 = sphere_dir[0][2] * scale

                    sphere_directors1[sphere_idx].set_xdata([sphere_position[0], sphere_position[0] + dir1[0]])
                    sphere_directors1[sphere_idx].set_ydata([sphere_position[1], sphere_position[1] + dir1[1]])

                    sphere_directors2[sphere_idx].set_xdata([sphere_position[0], sphere_position[0] + dir2[0]])
                    sphere_directors2[sphere_idx].set_ydata([sphere_position[1], sphere_position[1] + dir2[1]])        

                    sphere_directors3[sphere_idx].set_xdata([sphere_position[0], sphere_position[0] + dir3[0]])
                    sphere_directors3[sphere_idx].set_ydata([sphere_position[1], sphere_position[1] + dir3[1]])
                writer.grab_frame()



def plot_video_with_sphere_2D_obstacles(
    rods_history: Sequence[Dict],
    sphere_history: Sequence[Dict],
    video_name="video_2D.mp4",
    fps=60,
    step=1,
    vis2D=True,
    **kwargs,
):
    plt.rcParams.update({"font.size": 22})

    # 2d case <always 2d case for now>
    import matplotlib.animation as animation
    from matplotlib.patches import Circle

    # simulation time
    sim_time = np.array(sphere_history[0]["time"])

    # Sphere
    n_visualized_spheres = len(sphere_history)  # should be one for now
    # Sphere radius
    sphere_radii = [x["radius"] for x in sphere_history]
    # Sphere info
    print(sphere_history)
    sphere_history_unpacker = lambda sph_idx, t_idx: (
        sphere_history[sph_idx]["position"][t_idx],
        sphere_history[sph_idx]["radius"][t_idx],
    )
    # color mapping
    sphere_cmap = cm.get_cmap("Spectral", n_visualized_spheres)

    # Rod
    n_visualized_rods = len(rods_history)  # should be one for now
    # Rod info
    rod_history_unpacker = lambda rod_idx, t_idx: (
        rods_history[rod_idx]["position"][t_idx],
        rods_history[rod_idx]["radius"][t_idx],
    )
    # Rod center of mass
    com_history_unpacker = lambda rod_idx, t_idx: rods_history[rod_idx]["com"][time_idx]

    # video pre-processing
    print("plot scene visualization video")
    FFMpegWriter = animation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    dpi = kwargs.get("dpi", 100)

    xlim = kwargs.get("x_limits", (-1.0, 1.0))
    ylim = kwargs.get("y_limits", (-1.0, 1.0))
    zlim = kwargs.get("z_limits", (-0.05, 1.0))

    difference = lambda x: x[1] - x[0]
    max_axis_length = max(difference(xlim), difference(ylim))
    # The scaling factor from physical space to matplotlib space
    scaling_factor = (2 * 0.1) / max_axis_length  # Octopus head dimension
    scaling_factor *= 2.6e3  # Along one-axis
    # scaling_factor = 1.0

    fig = plt.figure(2, figsize=(10, 8), frameon=True, dpi=dpi)
    ax = fig.add_subplot(111)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    time_idx = 0
    rod_lines = [None for _ in range(n_visualized_rods)]
    rod_com_lines = [None for _ in range(n_visualized_rods)]
    rod_scatters = [None for _ in range(n_visualized_rods)]

    for rod_idx in range(n_visualized_rods):
        inst_position, inst_radius = rod_history_unpacker(rod_idx, time_idx)
        inst_position = 0.5 * (inst_position[..., 1:] + inst_position[..., :-1])
        rod_lines[rod_idx] = ax.plot(inst_position[0], inst_position[1], "r", lw=0.5)[0]
        inst_com = com_history_unpacker(rod_idx, time_idx)
        rod_com_lines[rod_idx] = ax.plot(inst_com[0], inst_com[1], "k--", lw=2.0)[0]

        rod_scatters[rod_idx] = ax.scatter(
            inst_position[0],
            inst_position[1],
            s=np.pi * (scaling_factor * inst_radius) ** 2,
        )

    sphere_artists = [None for _ in range(n_visualized_spheres)]
    for sphere_idx in range(n_visualized_spheres):
        sphere_position, sphere_radius = sphere_history_unpacker(sphere_idx, time_idx)
        sphere_artists[sphere_idx] = Circle(
            (sphere_position[0], sphere_position[1]),
            sphere_radius,
            color=sphere_cmap(sphere_idx),
        )
        ax.add_artist(sphere_artists[sphere_idx])

    ax.set_aspect("equal")
    video_name = "2D_" + video_name

    with writer.saving(fig, video_name, dpi):
        with plt.style.context("seaborn-whitegrid"):
            for time_idx in tqdm(range(0, sim_time.shape[0], int(step))):

                for rod_idx in range(n_visualized_rods):
                    inst_position, inst_radius = rod_history_unpacker(rod_idx, time_idx)
                    inst_position = 0.5 * (
                        inst_position[..., 1:] + inst_position[..., :-1]
                    )

                    rod_lines[rod_idx].set_xdata(inst_position[0])
                    rod_lines[rod_idx].set_ydata(inst_position[1])

                    com = com_history_unpacker(rod_idx, time_idx)
                    rod_com_lines[rod_idx].set_xdata(com[0])
                    rod_com_lines[rod_idx].set_ydata(com[1])

                    rod_scatters[rod_idx].set_offsets(inst_position[:2].T)
                    rod_scatters[rod_idx].set_sizes(
                        np.pi * (scaling_factor * inst_radius) ** 2
                    )
                
                for sphere_idx in range(n_visualized_spheres):
                    sphere_position, _ = sphere_history_unpacker(sphere_idx, time_idx)
                    sphere_artists[sphere_idx].center = (
                        sphere_position[0],
                        sphere_position[1],
                    )
                # sphere_position, _ = sphere_history_unpacker(0, time_idx)
                # ax.scatter(sphere_position[0]+0.05,sphere_position[1],)
                writer.grab_frame()
    plt.close(plt.gcf())


def plot_video_with_sphere_obstacles(
    rods_history: Sequence[Dict],
    sphere_history: Sequence[Dict],
    video_name="video.mp4",
    fps=60,
    step=1,
    vis2D=True,
    **kwargs,
):
    plt.rcParams.update({"font.size": 22})

    # 2d case <always 2d case for now>
    import matplotlib.animation as animation
    from matplotlib.patches import Circle
    from mpl_toolkits.mplot3d import proj3d, Axes3D

    # simulation time
    sim_time = np.array(sphere_history[0]["time"])

    # Sphere
    n_visualized_spheres = len(sphere_history)  # should be one for now
    # Sphere radius
    # sphere_radii = [x["radius"] for x in sphere_history]
    # Sphere info
    sphere_history_unpacker = lambda sph_idx, t_idx: (
        sphere_history[sph_idx]["position"][t_idx],
        sphere_history[sph_idx]["radius"][t_idx],
    )
    # color mapping
    sphere_cmap = cm.get_cmap("Spectral", n_visualized_spheres)

    # Rod
    n_visualized_rods = len(rods_history)  # should be one for now
    # Rod info
    rod_history_unpacker = lambda rod_idx, t_idx: (
        rods_history[rod_idx]["position"][t_idx],
        rods_history[rod_idx]["radius"][t_idx],
    )
    # Rod center of mass
    com_history_unpacker = lambda rod_idx, t_idx: rods_history[rod_idx]["com"][time_idx]

    # video pre-processing
    print("plot scene visualization video")
    FFMpegWriter = animation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata)
    dpi = kwargs.get("dpi", 100)

    xlim = kwargs.get("x_limits", (-1.0, 1.0))
    ylim = kwargs.get("y_limits", (-1.0, 1.0))
    zlim = kwargs.get("z_limits", (-0.0, 1.0))

    difference = lambda x: x[1] - x[0]
    max_axis_length = max(difference(xlim), difference(ylim))
    # The scaling factor from physical space to matplotlib space
    scaling_factor = (2 * 0.1) / max_axis_length  # Octopus head dimension
    scaling_factor *= 2.6e3  # Along one-axis
    # scaling_factor = 1.0

    fig = plt.figure(2, figsize=(10, 8), frameon=True, dpi=dpi)
    # ax = fig.add_subplot(111)
    ax = plt.axes(projection="3d")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.view_init(elev=30, azim=170)

    time_idx = 0
    rod_lines = [None for _ in range(n_visualized_rods)]
    rod_com_lines = [None for _ in range(n_visualized_rods)]
    rod_scatters = [None for _ in range(n_visualized_rods)]

    for rod_idx in range(n_visualized_rods):
        inst_position, inst_radius = rod_history_unpacker(rod_idx, time_idx)
        inst_position = 0.5 * (inst_position[..., 1:] + inst_position[..., :-1])
        # rod_lines[rod_idx] = ax.plot(inst_position[0], inst_position[1],inst_position[2], "r", lw=0.5)[0]
        inst_com = com_history_unpacker(rod_idx, time_idx)
        # rod_com_lines[rod_idx] = ax.plot(inst_com[0], inst_com[1],inst_com[2], "k--", lw=2.0)[0]

        rod_scatters[rod_idx] = ax.scatter(
            inst_position[2],
            inst_position[0],
            inst_position[1],
            s=np.pi * (scaling_factor * inst_radius) ** 2,
        )

    sphere_artists = [None for _ in range(n_visualized_spheres)]
    for sphere_idx in range(n_visualized_spheres):
        sphere_position, sphere_radius = sphere_history_unpacker(sphere_idx, time_idx)
        sphere_artists[sphere_idx] = ax.scatter(
            sphere_position[2],
            sphere_position[0],
            sphere_position[1],
            s=np.pi * (scaling_factor * sphere_radius) ** 2,
        )
        # sphere_radius,
        # color=sphere_cmap(sphere_idx),)
        ax.add_artist(sphere_artists[sphere_idx])

    # ax.set_aspect("equal")
    video_name = "3D_" + video_name
    
    with writer.saving(fig, video_name, dpi):
        with plt.style.context("seaborn-whitegrid"):
            for time_idx in tqdm(range(0, sim_time.shape[0], int(step))):

                for rod_idx in range(n_visualized_rods):
                    inst_position, inst_radius = rod_history_unpacker(rod_idx, time_idx)
                    inst_position = 0.5 * (
                        inst_position[..., 1:] + inst_position[..., :-1]
                    )

                    # rod_lines[rod_idx].set_xdata(inst_position[0])
                    # rod_lines[rod_idx].set_ydata(inst_position[1])
                    # rod_lines[rod_idx].set_zdata(inst_position[2])

                    com = com_history_unpacker(rod_idx, time_idx)
                    # rod_com_lines[rod_idx].set_xdata(com[0])
                    # rod_com_lines[rod_idx].set_ydata(com[1])
                    # rod_com_lines[rod_idx].set_zdata(com[2])

                    # rod_scatters[rod_idx].set_offsets(inst_position[:3].T)
                    rod_scatters[rod_idx]._offsets3d = (
                        inst_position[2],
                        inst_position[0],
                        inst_position[1],
                    )

                    rod_scatters[rod_idx].set_sizes(
                        np.pi * (scaling_factor * inst_radius) ** 2 * 0.1
                    )

                for sphere_idx in range(n_visualized_spheres):
                    sphere_position, _ = sphere_history_unpacker(sphere_idx, time_idx)
                    # sphere_artists[sphere_idx].center = (
                    #     sphere_position[0],
                    #     sphere_position[1],
                    #     sphere_position[2],
                    # )
                    # sphere_artists[sphere_idx].set_offsets(sphere_position[:3].T)
                    sphere_artists[sphere_idx]._offsets3d = (
                        sphere_position[2],
                        sphere_position[0],
                        sphere_position[1],
                    )

                writer.grab_frame()

    # Be a good boy and close figures
    # https://stackoverflow.com/a/37451036
    # plt.close(fig) alone does not suffice
    # See https://github.com/matplotlib/matplotlib/issues/8560/
    plt.close(plt.gcf())


