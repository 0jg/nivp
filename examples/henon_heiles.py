import marimo

__generated_with = "0.17.0"
app = marimo.App()


@app.cell
def _():
    import torch
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.rcParams['font.family'] = 'Times New Roman'
    matplotlib.rcParams['font.size'] = 11
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['axes.linewidth'] = 0.8
    matplotlib.rcParams['xtick.major.width'] = 0.8
    matplotlib.rcParams['ytick.major.width'] = 0.8
    return plt, torch, tqdm


@app.cell
def _():
    import json
    import os
    import sys

    # Add parent directory to path to import modules
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    from helpers import (
        DATA_TYPE,
        SimpleMLP,
        CoupledOptimiser,
        expected_path_tensor,
        ensure_run_dirs,
        second_order_dynamics,
    )
    from systems import HenonHeiles
    return (
        CoupledOptimiser,
        DATA_TYPE,
        HenonHeiles,
        SimpleMLP,
        second_order_dynamics,
        ensure_run_dirs,
        expected_path_tensor,
        json,
        os,
    )


@app.cell
def controls():
    # Chaotic trajectory with E=1/6
    params_chaotic = {
        "activation": "sechlu",
        "hidden": 64,
        "epochs": 100_000,
        "batch_size": 32,
        "lr": 3e-3,
        "T_MIN": 0.0,
        "T_MAX": 15.0,
        "TIMESTEP": 0.05,
        "ENERGY": 1/6,
        "x0": 0.0,
        "y0": 0.57,
        "vx0": 0.10,
        "vy0": 0.057,
        "ODE_REGULARISER": 1.0,
        "POSITION_REGULARISER": 1.0,
        "VELOCITY_REGULARISER": 1.0,
        "JOB_ID": 1,
    }

    # Quasi-periodic trajectory with E=1/12
    params_quasiperiodic = {  # noqa: F841
        "activation": "sechlu",
        "hidden": 64,
        "epochs": 200,
        "batch_size": 128,
        "lr": 3e-3,
        "T_MIN": 0.0,
        "T_MAX": 15.0,
        "TIMESTEP": 0.01,
        "ENERGY": 1/12,
        "x0": 0.0,
        "y0": 0.40,
        "vx0": 0.05,
        "vy0": 0.041,
        "ODE_REGULARISER": 1.0,
        "POSITION_REGULARISER": 1.0,
        "VELOCITY_REGULARISER": 1.0,
        "JOB_ID": 2,
    }

    # Select which trajectory to train
    # Uncomment to switch: params = params_quasiperiodic
    params = params_chaotic
    return (params,)


@app.cell
def dataset(DATA_TYPE, HenonHeiles, expected_path_tensor, params, torch):
    T_MIN, T_MAX, DT = params["T_MIN"], params["T_MAX"], params["TIMESTEP"]
    grid = torch.linspace(T_MIN, T_MAX, int(round((T_MAX-T_MIN)/DT))+1, dtype=DATA_TYPE)
    t = grid.unsqueeze(0).unsqueeze(2).requires_grad_(True)
    t_dataset = torch.utils.data.TensorDataset(t)
    domain = torch.utils.data.DataLoader(dataset=t_dataset, batch_size=params["batch_size"], shuffle=True)

    init_position = torch.tensor([params["x0"], params["y0"]], dtype=DATA_TYPE)
    init_velocity = torch.tensor([params["vx0"], params["vy0"]], dtype=DATA_TYPE)

    system = HenonHeiles()
    expected = expected_path_tensor(system, init_position, init_velocity, T_MIN, T_MAX, DT)
    return domain, expected, t, system, init_position, init_velocity


@app.cell
def train_loop(
    CoupledOptimiser,
    SimpleMLP,
    domain,
    ensure_run_dirs,
    expected,
    init_position,
    init_velocity,
    json,
    os,
    params,
    plt,
    second_order_dynamics,
    system,
    t,
    torch,
    tqdm,
):
    # Build models
    model_x = SimpleMLP(in_features=1, hidden=params["hidden"], out_features=1, activation=params["activation"])
    model_y = SimpleMLP(in_features=1, hidden=params["hidden"], out_features=1, activation=params["activation"])
    opt_x = torch.optim.Adam(model_x.parameters(), lr=params["lr"])
    opt_y = torch.optim.Adam(model_y.parameters(), lr=params["lr"])
    optimiser = CoupledOptimiser(opt_x, opt_y)

    def train():
        EPOCHS = params["epochs"]
        loss_hist = []
        for epoch in tqdm(range(EPOCHS)):
            for (t_batch,) in domain:
                px = model_x(t_batch).requires_grad_(True)
                py = model_y(t_batch).requires_grad_(True)
                positions = torch.cat([px, py], dim=2)
                velocities, _, residuals = second_order_dynamics(system, positions, t_batch)
                target_pos = init_position.to(positions).unsqueeze(0)
                target_vel = init_velocity.to(positions).unsqueeze(0)

                l2 = torch.nn.MSELoss()
                pos_loss = l2(positions[:, 0, :], target_pos)
                vel_loss = l2(velocities[:, 0, :], target_vel)
                ode_loss = l2(residuals, torch.zeros_like(residuals))

                overall = (
                    params["ODE_REGULARISER"] * ode_loss
                    + params["POSITION_REGULARISER"] * pos_loss
                    + params["VELOCITY_REGULARISER"] * vel_loss
                )
                loss_hist.append(overall.item())

                optimiser.zero_grad()
                overall.backward()
                optimiser.step()

            if (epoch % 500 == 0) or (epoch == EPOCHS-1):
                fig = plt.figure(figsize=(12, 5), dpi=150)
                fig.patch.set_facecolor('white')

                # Create GridSpec for legend at top with 2 rows, 2 cols (no loss subplot)
                gs = fig.add_gridspec(2, 2, height_ratios=[0.1, 1], hspace=0.3, wspace=0.3, top=0.92)

                # Legend axis at top (spanning both columns)
                ax_legend = fig.add_subplot(gs[0, :])
                ax_legend.axis('off')

                tt = t[0,:,0].detach().cpu().numpy()
                px_np = positions.detach().cpu().numpy()[0,:,0]
                py_np = positions.detach().cpu().numpy()[0,:,1]
                expected_x = expected.detach().cpu().numpy()[0,:,0]
                expected_y = expected.detach().cpu().numpy()[0,:,1]

                # Create dummy lines for legend
                line_pinn, = ax_legend.plot([], [], color='black', linewidth=2, label='Neural IVP')
                line_rk4, = ax_legend.plot([], [], color='lightgrey', linewidth=2.5, label='RK4')

                ax_legend.legend(
                    handles=[line_pinn, line_rk4],
                    loc='upper center',
                    ncol=2,
                    frameon=False,
                    fontsize=11,
                    bbox_to_anchor=(0.5, 0.5)
                )

                # x(t) trajectory
                ax1 = fig.add_subplot(gs[1, 0])
                ax1.plot(tt, px_np, color='black', linewidth=2, label='Neural IVP')
                ax1.plot(tt, expected_x, color='lightgrey', linewidth=2.5, label='RK4', zorder=-1)
                ax1.set_xlabel(r'$t$', fontsize=12)
                ax1.set_ylabel(r'$x(t)$', fontsize=12)
                ax1.spines['top'].set_visible(False)
                ax1.spines['right'].set_visible(False)
                ax1.tick_params(direction='in', which='both', left=True, bottom=True)
                ax1.tick_params(labelbottom=True, labelleft=True)

                # y(t) trajectory
                ax2 = fig.add_subplot(gs[1, 1])
                ax2.plot(tt, py_np, color='black', linewidth=2, label='Neural IVP')
                ax2.plot(tt, expected_y, color='lightgrey', linewidth=2.5, label='RK4', zorder=-1)
                ax2.set_xlabel(r'$t$', fontsize=12)
                ax2.set_ylabel(r'$y(t)$', fontsize=12)
                ax2.spines['top'].set_visible(False)
                ax2.spines['right'].set_visible(False)
                ax2.tick_params(direction='in', which='both', left=True, bottom=True)
                ax2.tick_params(labelbottom=True, labelleft=True)

                plt.show()

                # Phase space as separate square plot
                fig_phase = plt.figure(figsize=(6, 6), dpi=150)
                fig_phase.patch.set_facecolor('white')
                ax_phase = fig_phase.add_subplot(111)
                ax_phase.plot(px_np, py_np, color='black', linewidth=2, label='Neural IVP')
                ax_phase.plot(expected_x, expected_y, color='lightgrey', linewidth=2.5, label='RK4', zorder=-1)
                ax_phase.set_xlabel(r'$x(t)$', fontsize=12)
                ax_phase.set_ylabel(r'$y(x(t))$', fontsize=12)
                ax_phase.spines['top'].set_visible(False)
                ax_phase.spines['right'].set_visible(False)
                ax_phase.tick_params(direction='in', which='both', left=True, bottom=True)
                ax_phase.tick_params(labelbottom=True, labelleft=True)
                plt.show()

                # Loss history as separate plot
                fig_loss = plt.figure(figsize=(8, 5), dpi=150)
                fig_loss.patch.set_facecolor('white')
                ax_loss = fig_loss.add_subplot(111)
                ax_loss.semilogy(loss_hist, linewidth=2, color='black')
                ax_loss.set_xlabel(r'Batch', fontsize=12)
                ax_loss.set_ylabel(r'Loss', fontsize=12)
                ax_loss.spines['top'].set_visible(False)
                ax_loss.spines['right'].set_visible(False)
                ax_loss.tick_params(direction='in', which='both', left=True, bottom=True)
                ax_loss.tick_params(labelbottom=True, labelleft=True)
                plt.show()

        return model_x, model_y, loss_hist, positions

    model_x, model_y, loss_hist, positions_final = train()

    # Save results to examples/outputs
    output_base = os.path.join(os.path.dirname(__file__), "outputs")
    paths = ensure_run_dirs(output_base, params, system_name="henon_heiles")

    # Save models
    mx_path = os.path.join(paths["models"], f'{params["JOB_ID"]}_x.pth')
    my_path = os.path.join(paths["models"], f'{params["JOB_ID"]}_y.pth')
    torch.save(model_x, mx_path)
    torch.save(model_y, my_path)

    # Save loss history
    with open(os.path.join(paths["paths"], "loss_history.json"), "w") as f:
        json.dump(loss_hist, f)

    # Generate and save final figures
    tt = t[0,:,0].detach().cpu().numpy()
    px_np = positions_final.detach().cpu().numpy()[0,:,0]
    py_np = positions_final.detach().cpu().numpy()[0,:,1]
    expected_x = expected.detach().cpu().numpy()[0,:,0]
    expected_y = expected.detach().cpu().numpy()[0,:,1]

    # Save trajectories figure
    fig = plt.figure(figsize=(12, 5), dpi=150)
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(2, 2, height_ratios=[0.1, 1], hspace=0.3, wspace=0.3, top=0.92)

    ax_legend = fig.add_subplot(gs[0, :])
    ax_legend.axis('off')
    line_pinn, = ax_legend.plot([], [], color='black', linewidth=2, label='Neural IVP')
    line_rk4, = ax_legend.plot([], [], color='lightgrey', linewidth=2.5, label='RK4')
    ax_legend.legend(handles=[line_pinn, line_rk4], loc='upper center', ncol=2, 
                     frameon=False, fontsize=11, bbox_to_anchor=(0.5, 0.5))

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(tt, px_np, color='black', linewidth=2, label='Neural IVP')
    ax1.plot(tt, expected_x, color='lightgrey', linewidth=2.5, label='RK4', zorder=-1)
    ax1.set_xlabel(r'$t$', fontsize=12)
    ax1.set_ylabel(r'$x(t)$', fontsize=12)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.tick_params(direction='in', which='both', left=True, bottom=True)
    ax1.tick_params(labelbottom=True, labelleft=True)

    ax2 = fig.add_subplot(gs[1, 1])
    ax2.plot(tt, py_np, color='black', linewidth=2, label='Neural IVP')
    ax2.plot(tt, expected_y, color='lightgrey', linewidth=2.5, label='RK4', zorder=-1)
    ax2.set_xlabel(r'$t$', fontsize=12)
    ax2.set_ylabel(r'$y(t)$', fontsize=12)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.tick_params(direction='in', which='both', left=True, bottom=True)
    ax2.tick_params(labelbottom=True, labelleft=True)
    
    traj_path = os.path.join(paths["figures"], "trajectories.png")
    fig.savefig(traj_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Save phase space figure
    fig_phase = plt.figure(figsize=(6, 6), dpi=150)
    fig_phase.patch.set_facecolor('white')
    ax_phase = fig_phase.add_subplot(111)
    ax_phase.plot(px_np, py_np, color='black', linewidth=2, label='Neural IVP')
    ax_phase.plot(expected_x, expected_y, color='lightgrey', linewidth=2.5, label='RK4', zorder=-1)
    ax_phase.set_xlabel(r'$x(t)$', fontsize=12)
    ax_phase.set_ylabel(r'$y(x(t))$', fontsize=12)
    ax_phase.spines['top'].set_visible(False)
    ax_phase.spines['right'].set_visible(False)
    ax_phase.tick_params(direction='in', which='both', left=True, bottom=True)
    ax_phase.tick_params(labelbottom=True, labelleft=True)
    
    phase_path = os.path.join(paths["figures"], "phase_space.png")
    fig_phase.savefig(phase_path, dpi=150, bbox_inches='tight')
    plt.close(fig_phase)

    # Save loss history figure
    fig_loss = plt.figure(figsize=(8, 5), dpi=150)
    fig_loss.patch.set_facecolor('white')
    ax_loss = fig_loss.add_subplot(111)
    ax_loss.semilogy(loss_hist, linewidth=2, color='black')
    ax_loss.set_xlabel(r'Batch', fontsize=12)
    ax_loss.set_ylabel(r'Loss', fontsize=12)
    ax_loss.spines['top'].set_visible(False)
    ax_loss.spines['right'].set_visible(False)
    ax_loss.tick_params(direction='in', which='both', left=True, bottom=True)
    ax_loss.tick_params(labelbottom=True, labelleft=True)
    
    loss_path = os.path.join(paths["figures"], "loss_history.png")
    fig_loss.savefig(loss_path, dpi=150, bbox_inches='tight')
    plt.close(fig_loss)

    print("**Saved**:")
    print(f"- Model X: {mx_path}")
    print(f"- Model Y: {my_path}")
    print(f"- Loss history: {os.path.join(paths['paths'], 'loss_history.json')}")
    print(f"- Trajectories: {traj_path}")
    print(f"- Phase space: {phase_path}")
    print(f"- Loss plot: {loss_path}")
    return


if __name__ == "__main__":
    app.run()
