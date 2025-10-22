import marimo

__generated_with = "0.17.0"
app = marimo.App()


@app.cell
def _():
    import torch
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import matplotlib
    import numpy as np

    # Set up device: MPS (Apple Silicon) > CUDA > CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    matplotlib.rcParams['font.family'] = 'Times New Roman'
    matplotlib.rcParams['font.size'] = 11
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['axes.linewidth'] = 0.8
    matplotlib.rcParams['xtick.major.width'] = 0.8
    matplotlib.rcParams['ytick.major.width'] = 0.8
    return device, np, plt, torch, tqdm


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
        expected_path_tensor_first_order,
        ensure_run_dirs,
        first_order_dynamics,
    )
    from systems import LandauLifschitz
    return (
        CoupledOptimiser,
        DATA_TYPE,
        LandauLifschitz,
        SimpleMLP,
        ensure_run_dirs,
        expected_path_tensor_first_order,
        first_order_dynamics,
        json,
        os,
    )


@app.cell
def controls():
    # Landau-Lifschitz equation with H = H0 e_z
    # Initial condition: M(0) = e_x (i.e., mx(0)=1, my(0)=0, mz(0)=0)
    params = {
        "activation": "sechlu",
        "hidden": 64,
        "epochs": 30_000,
        "batch_size": 32,
        "lr": 1e-3,
        "T_MIN": 0.0,
        "T_MAX": 20.0,
        "TIMESTEP": 0.01,
        "mx0": 1.0,
        "my0": 0.0,
        "mz0": 0.0,
        "alpha": 0.25,
        "H0": 1.0,
        "ODE_REGULARISER": 1.0,
        "INITIAL_CONDITION_REGULARISER": 1.0,
        "JOB_ID": 1,
    }
    return (params,)


@app.cell
def dataset(
    DATA_TYPE,
    LandauLifschitz,
    expected_path_tensor_first_order,
    params,
    torch,
):
    T_MIN, T_MAX, DT = params["T_MIN"], params["T_MAX"], params["TIMESTEP"]
    grid = torch.linspace(T_MIN, T_MAX, int(round((T_MAX-T_MIN)/DT))+1, dtype=DATA_TYPE)
    t = grid.unsqueeze(0).unsqueeze(2).requires_grad_(True)
    t_dataset = torch.utils.data.TensorDataset(t)
    domain = torch.utils.data.DataLoader(dataset=t_dataset, batch_size=params["batch_size"], shuffle=True)

    init_state = torch.tensor([params["mx0"], params["my0"], params["mz0"]], dtype=DATA_TYPE)

    system = LandauLifschitz(alpha=params["alpha"], H0=params["H0"])
    expected = expected_path_tensor_first_order(system, init_state, T_MIN, T_MAX, DT)
    return domain, expected, init_state, system, t


@app.cell
def analytical_solution(np, params):
    def compute_analytical(time_array):
        """
        Compute analytical solution for LL equation with H = H_0 e_z.
    
        In physical units,
            M_x(t) = sech(alpha tau) cos(tau)
            M_y(t) = sech(alpha tau) sin(tau)
            M_z(t) = tanh(alpha tau)
        """
        alpha = params["alpha"]
        tau = time_array

        mx_analytical = (1.0 / np.cosh(alpha * tau)) * np.cos(tau)
        my_analytical = (1.0 / np.cosh(alpha * tau)) * np.sin(tau)
        mz_analytical = np.tanh(alpha * tau)

        return mx_analytical, my_analytical, mz_analytical
    return (compute_analytical,)


@app.cell
def train_loop(
    CoupledOptimiser,
    SimpleMLP,
    compute_analytical,
    device,
    domain,
    ensure_run_dirs,
    expected,
    first_order_dynamics,
    init_state,
    json,
    np,
    os,
    params,
    plt,
    system,
    t,
    torch,
    tqdm,
):
    # Build models - one for each component of magnetization
    model_mx = SimpleMLP(in_features=1, hidden=params["hidden"], out_features=1, activation=params["activation"]).to(device)
    model_my = SimpleMLP(in_features=1, hidden=params["hidden"], out_features=1, activation=params["activation"]).to(device)
    model_mz = SimpleMLP(in_features=1, hidden=params["hidden"], out_features=1, activation=params["activation"]).to(device)

    opt_mx = torch.optim.Adam(model_mx.parameters(), lr=params["lr"])
    opt_my = torch.optim.Adam(model_my.parameters(), lr=params["lr"])
    opt_mz = torch.optim.Adam(model_mz.parameters(), lr=params["lr"])
    optimiser = CoupledOptimiser(opt_mx, opt_my, opt_mz)

    def train():
        EPOCHS = params["epochs"]
        loss_hist = []
        for epoch in tqdm(range(EPOCHS)):
            for (t_batch,) in domain:
                t_batch = t_batch.to(device)
                mx = model_mx(t_batch).requires_grad_(True)
                my = model_my(t_batch).requires_grad_(True)
                mz = model_mz(t_batch).requires_grad_(True)
                M = torch.cat([mx, my, mz], dim=2)

                _, residuals = first_order_dynamics(system, M, t_batch)

                target_state = init_state.to(device).unsqueeze(0)

                l2 = torch.nn.MSELoss()

                # Initial condition losses
                ic_loss = l2(M[:, 0, :], target_state)

                # ODE residual loss
                ode_loss = l2(residuals, torch.zeros_like(residuals))

                overall = (
                    params["ODE_REGULARISER"] * ode_loss
                    + params["INITIAL_CONDITION_REGULARISER"] * ic_loss
                )
                loss_hist.append(overall.item())

                optimiser.zero_grad()
                overall.backward()
                optimiser.step()

            if (epoch % 1000 == 0) or (epoch == EPOCHS-1):
                fig = plt.figure(figsize=(10, 5), dpi=150)
                fig.patch.set_facecolor('white')

                # Create GridSpec for trajectories with residuals
                # height_ratios: legend (0.06) : main_plots (0.75) : residuals (0.25)
                gs = fig.add_gridspec(3, 3, height_ratios=[0.06, .75, 0.25], hspace=0.15, wspace=0.25, 
                                      left=0.07, right=0.98, top=0.97, bottom=0.07)

                # Legend axis at top (spanning all columns)
                ax_legend = fig.add_subplot(gs[0, :])
                ax_legend.axis('off')

                tt = t[0,:,0].detach().cpu().numpy()
                mx_np = M.detach().cpu().numpy()[0,:,0]
                my_np = M.detach().cpu().numpy()[0,:,1]
                mz_np = M.detach().cpu().numpy()[0,:,2]
                expected_mx = expected.detach().cpu().numpy()[0,:,0]
                expected_my = expected.detach().cpu().numpy()[0,:,1]
                expected_mz = expected.detach().cpu().numpy()[0,:,2]

                # Compute analytical solution
                mx_analytical, my_analytical, mz_analytical = compute_analytical(tt)

                line_pinn, = ax_legend.plot([], [], color='black', linewidth=2, label='Neural IVP')
                line_rk4, = ax_legend.plot([], [], color='darkgrey', linewidth=2, linestyle='--', label='RK4')
                line_analytical, = ax_legend.plot([], [], color='lightgrey', linewidth=2, label='Analytical')

                ax_legend.legend(
                    handles=[line_pinn, line_rk4, line_analytical],
                    loc='upper center',
                    ncol=3,
                    frameon=False,
                    fontsize=11,
                    bbox_to_anchor=(0.5, 0.5)
                )

                # mx plots
                ax1 = fig.add_subplot(gs[1, 0])
                ax1.plot(tt, mx_analytical, color='lightgrey', linewidth=2, zorder=1)
                ax1.plot(tt, expected_mx, color='darkgrey', linewidth=2, linestyle='--', zorder=2)
                ax1.plot(tt, mx_np, color='black', linewidth=2, zorder=3)
                ax1.set_ylabel(r'$m_x(\tau)$', fontsize=12)
                ax1.spines['top'].set_visible(False)
                ax1.spines['right'].set_visible(False)
                ax1.tick_params(direction='in', which='both', left=True, bottom=True)
                ax1.set_xticklabels([])

                ax1_res = fig.add_subplot(gs[2, 0])
                ax1_res.plot(tt, mx_np - mx_analytical, color='black', linewidth=2, zorder=3)
                ax1_res.plot(tt, expected_mx - mx_analytical, color='darkgrey', linewidth=2, linestyle='--', zorder=2)
                ax1_res.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
                ax1_res.set_xlabel(r'$\tau $', fontsize=11)
                ax1_res.set_ylabel('Residual', fontsize=10)
                ax1_res.spines['top'].set_visible(False)
                ax1_res.spines['right'].set_visible(False)
                ax1_res.tick_params(direction='in', which='both', left=True, bottom=True, labelsize=9)

                # my plots
                ax2 = fig.add_subplot(gs[1, 1])
                ax2.plot(tt, my_analytical, color='lightgrey', linewidth=2, zorder=1)
                ax2.plot(tt, expected_my, color='darkgrey', linewidth=2, linestyle='--', zorder=2)
                ax2.plot(tt, my_np, color='black', linewidth=2, zorder=3)
                ax2.set_ylabel(r'$m_y(\tau)$', fontsize=12)
                ax2.spines['top'].set_visible(False)
                ax2.spines['right'].set_visible(False)
                ax2.tick_params(direction='in', which='both', left=True, bottom=True)
                ax2.set_xticklabels([])

                ax2_res = fig.add_subplot(gs[2, 1])
                ax2_res.plot(tt, my_np - my_analytical, color='black', linewidth=2, zorder=3)
                ax2_res.plot(tt, expected_my - my_analytical, color='darkgrey', linewidth=2, linestyle='--', zorder=2)
                ax2_res.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
                ax2_res.set_xlabel(r'$\tau $', fontsize=11)
                ax2_res.set_ylabel('Residual', fontsize=10)
                ax2_res.spines['top'].set_visible(False)
                ax2_res.spines['right'].set_visible(False)
                ax2_res.tick_params(direction='in', which='both', left=True, bottom=True, labelsize=9)

                # mz plots
                ax3 = fig.add_subplot(gs[1, 2])
                ax3.plot(tt, mz_analytical, color='lightgrey', linewidth=2, zorder=1)
                ax3.plot(tt, expected_mz, color='darkgrey', linewidth=2, linestyle='--', zorder=2)
                ax3.plot(tt, mz_np, color='black', linewidth=2, zorder=3)
                ax3.set_ylabel(r'$m_z(\tau)$', fontsize=12)
                ax3.spines['top'].set_visible(False)
                ax3.spines['right'].set_visible(False)
                ax3.tick_params(direction='in', which='both', left=True, bottom=True)
                ax3.set_xticklabels([])

                ax3_res = fig.add_subplot(gs[2, 2])
                ax3_res.plot(tt, mz_np - mz_analytical, color='black', linewidth=2, zorder=3)
                ax3_res.plot(tt, expected_mz - mz_analytical, color='darkgrey', linewidth=2, linestyle='--', zorder=2)
                ax3_res.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
                ax3_res.set_xlabel(r'$\tau $', fontsize=11)
                ax3_res.set_ylabel('Residual', fontsize=10)
                ax3_res.spines['top'].set_visible(False)
                ax3_res.spines['right'].set_visible(False)
                ax3_res.tick_params(direction='in', which='both', left=True, bottom=True, labelsize=9)

                plt.show()

                # 3D trajectory in phase space
                from mpl_toolkits.mplot3d import Axes3D
                fig_3d = plt.figure(figsize=(7, 7), dpi=150)
                fig_3d.patch.set_facecolor('white')
                ax_3d = fig_3d.add_subplot(111, projection='3d')

                # Plot trajectories with correct z-ordering
                ax_3d.plot(mx_analytical, my_analytical, mz_analytical, color='lightgrey', 
                          linewidth=2, label='Analytical', zorder=1)
                ax_3d.plot(expected_mx, expected_my, expected_mz, color='darkgrey', linewidth=2, 
                          linestyle='--', label='RK4', zorder=2)
                ax_3d.plot(mx_np, my_np, mz_np, color='black', linewidth=2, label='Neural IVP', zorder=3)

                # Set axis limits first to ensure proper scaling
                max_range = 1.2
                ax_3d.set_xlim([-max_range, max_range])
                ax_3d.set_ylim([-max_range, max_range])
                ax_3d.set_zlim([-max_range, max_range])

                # Equal aspect ratio
                ax_3d.set_box_aspect([1,1,1])

                # Set viewing angle for better visualisation of all axes
                ax_3d.view_init(elev=25, azim=240)

                # Mark initial and final points
                ax_3d.scatter([mx_np[0]], [my_np[0]], [mz_np[0]], color='black', s=80, 
                             marker='o', zorder=10, edgecolors='black', linewidth=1)
                ax_3d.scatter([mx_np[-1]], [my_np[-1]], [mz_np[-1]], color='black', s=80, 
                             marker='s', zorder=10, edgecolors='black', linewidth=1, facecolors='black')

                # Add text annotations with white background
                ax_3d.text(mx_np[0], my_np[0], mz_np[0], '  Start', fontsize=9, 
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.8))
                ax_3d.text(mx_np[-1], my_np[-1], mz_np[-1], '  End', fontsize=9,
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.8))

                # Style the axes
                ax_3d.set_xlabel(r'$m_x$', fontsize=12, labelpad=8)
                ax_3d.set_ylabel(r'$m_y$', fontsize=12, labelpad=8)
                ax_3d.set_zlabel(r'$m_z$', fontsize=12, labelpad=8, rotation=180)

                # Remove grid
                ax_3d.grid(False)

                # Set pane colors to white
                ax_3d.xaxis.pane.fill = False
                ax_3d.yaxis.pane.fill = False
                ax_3d.zaxis.pane.fill = False

                # Set pane edge colors to light gray
                ax_3d.xaxis.pane.set_edgecolor('lightgray')
                ax_3d.yaxis.pane.set_edgecolor('lightgray')
                ax_3d.zaxis.pane.set_edgecolor('lightgray')

                # Make panes more subtle
                ax_3d.xaxis.pane.set_alpha(0.25)
                ax_3d.yaxis.pane.set_alpha(0.25)
                ax_3d.zaxis.pane.set_alpha(0.25)

                # Improve tick styling
                ax_3d.tick_params(axis='x', labelsize=10, pad=3)
                ax_3d.tick_params(axis='y', labelsize=10, pad=3)
                ax_3d.tick_params(axis='z', labelsize=10, pad=3)

                # Legend matching the style of 2D plots
                ax_3d.legend(loc='upper center', frameon=False, fontsize=11, ncol=3,
                            bbox_to_anchor=(0.5, 1.05))

                plt.tight_layout()
                plt.show()                # Loss history
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

        return model_mx, model_my, model_mz, loss_hist, M

    model_mx, model_my, model_mz, loss_hist, M_final = train()

    # Save results to examples/outputs
    output_base = os.path.join(os.path.dirname(__file__), "outputs")
    paths = ensure_run_dirs(output_base, params, system_name="landau_lifshitz")

    # Save models
    mx_path = os.path.join(paths["models"], f'{params["JOB_ID"]}_mx.pth')
    my_path = os.path.join(paths["models"], f'{params["JOB_ID"]}_my.pth')
    mz_path = os.path.join(paths["models"], f'{params["JOB_ID"]}_mz.pth')
    torch.save(model_mx, mx_path)
    torch.save(model_my, my_path)
    torch.save(model_mz, mz_path)

    # Save loss history
    with open(os.path.join(paths["paths"], "loss_history.json"), "w") as f:
        json.dump(loss_hist, f)

    # Generate and save final figures
    tt = t[0,:,0].detach().cpu().numpy()
    mx_np = M_final.detach().cpu().numpy()[0,:,0]
    my_np = M_final.detach().cpu().numpy()[0,:,1]
    mz_np = M_final.detach().cpu().numpy()[0,:,2]
    expected_mx = expected.detach().cpu().numpy()[0,:,0]
    expected_my = expected.detach().cpu().numpy()[0,:,1]
    expected_mz = expected.detach().cpu().numpy()[0,:,2]

    # Compute analytical solution
    mx_analytical, my_analytical, mz_analytical = compute_analytical(tt)

    # Save trajectories figure
    fig = plt.figure(figsize=(10, 5), dpi=150)
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(3, 3, height_ratios=[0.06, 0.75, 0.25], hspace=0.15, wspace=0.25,
                          left=0.07, right=0.98, top=0.97, bottom=0.07)

    # Legend
    ax_legend = fig.add_subplot(gs[0, :])
    ax_legend.axis('off')
    line_pinn, = ax_legend.plot([], [], color='black', linewidth=2, label='Neural IVP')
    line_rk4, = ax_legend.plot([], [], color='darkgrey', linewidth=2, linestyle='--', label='RK4')
    line_analytical, = ax_legend.plot([], [], color='lightgrey', linewidth=2, label='Analytical')
    ax_legend.legend(handles=[line_pinn, line_rk4, line_analytical], loc='upper center', ncol=3, 
                     frameon=False, fontsize=11, bbox_to_anchor=(0.5, 0.5))

    # mx plots
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(tt, mx_analytical, color='lightgrey', linewidth=2, zorder=1)
    ax1.plot(tt, expected_mx, color='darkgrey', linewidth=2, linestyle='--', zorder=2)
    ax1.plot(tt, mx_np, color='black', linewidth=2, zorder=3)
    ax1.set_ylabel(r'$m_x(\tau)$', fontsize=12)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.tick_params(direction='in', which='both', left=True, bottom=True)
    ax1.set_xticklabels([])

    ax1_res = fig.add_subplot(gs[2, 0])
    ax1_res.plot(tt, mx_np - mx_analytical, color='black', linewidth=2, zorder=3)
    ax1_res.plot(tt, expected_mx - mx_analytical, color='darkgrey', linewidth=2, linestyle='--', zorder=2)
    ax1_res.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax1_res.set_xlabel(r'$\tau $', fontsize=11)
    ax1_res.set_ylabel('Residual', fontsize=10)
    ax1_res.spines['top'].set_visible(False)
    ax1_res.spines['right'].set_visible(False)
    ax1_res.tick_params(direction='in', which='both', left=True, bottom=True, labelsize=9)

    # my plots
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.plot(tt, my_analytical, color='lightgrey', linewidth=2, zorder=1)
    ax2.plot(tt, expected_my, color='darkgrey', linewidth=2, linestyle='--', zorder=2)
    ax2.plot(tt, my_np, color='black', linewidth=2, zorder=3)
    ax2.set_ylabel(r'$m_y(\tau)$', fontsize=12)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.tick_params(direction='in', which='both', left=True, bottom=True)
    ax2.set_xticklabels([])

    ax2_res = fig.add_subplot(gs[2, 1])
    ax2_res.plot(tt, my_np - my_analytical, color='black', linewidth=2, zorder=3)
    ax2_res.plot(tt, expected_my - my_analytical, color='darkgrey', linewidth=2, linestyle='--', zorder=2)
    ax2_res.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax2_res.set_xlabel(r'$\tau $', fontsize=11)
    ax2_res.set_ylabel('Residual', fontsize=10)
    ax2_res.spines['top'].set_visible(False)
    ax2_res.spines['right'].set_visible(False)
    ax2_res.tick_params(direction='in', which='both', left=True, bottom=True, labelsize=9)

    # mz plots
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.plot(tt, mz_analytical, color='lightgrey', linewidth=2, zorder=1)
    ax3.plot(tt, expected_mz, color='darkgrey', linewidth=2, linestyle='--', zorder=2)
    ax3.plot(tt, mz_np, color='black', linewidth=2, zorder=3)
    ax3.set_ylabel(r'$m_z(\tau)$', fontsize=12)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.tick_params(direction='in', which='both', left=True, bottom=True)
    ax3.set_xticklabels([])

    ax3_res = fig.add_subplot(gs[2, 2])
    ax3_res.plot(tt, mz_np - mz_analytical, color='black', linewidth=2, zorder=3)
    ax3_res.plot(tt, expected_mz - mz_analytical, color='darkgrey', linewidth=2, linestyle='--', zorder=2)
    ax3_res.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax3_res.set_xlabel(r'$\tau $', fontsize=11)
    ax3_res.set_ylabel('Residual', fontsize=10)
    ax3_res.spines['top'].set_visible(False)
    ax3_res.spines['right'].set_visible(False)
    ax3_res.tick_params(direction='in', which='both', left=True, bottom=True, labelsize=9)

    # Compute and print MSE for each component
    mse_mx_pinn = np.mean((mx_np - mx_analytical) ** 2)
    mse_my_pinn = np.mean((my_np - my_analytical) ** 2)
    mse_mz_pinn = np.mean((mz_np - mz_analytical) ** 2)

    mse_mx_rk4 = np.mean((expected_mx - mx_analytical) ** 2)
    mse_my_rk4 = np.mean((expected_my - my_analytical) ** 2)
    mse_mz_rk4 = np.mean((expected_mz - mz_analytical) ** 2)

    print("\nMean Squared Error vs Analytical Solution:")
    print(f"Neural IVP:  mx={mse_mx_pinn:.2e}  my={mse_my_pinn:.2e}  mz={mse_mz_pinn:.2e}")
    print(f"RK4:         mx={mse_mx_rk4:.2e}  my={mse_my_rk4:.2e}  mz={mse_mz_rk4:.2e}")

    traj_path = os.path.join(paths["figures"], "trajectories.pdf")
    fig.savefig(traj_path, dpi=150, bbox_inches='tight')
    plt.close(fig)    # Save 3D trajectory
    from mpl_toolkits.mplot3d import Axes3D
    fig_3d = plt.figure(figsize=(7, 7), dpi=150)
    fig_3d.patch.set_facecolor('white')
    ax_3d = fig_3d.add_subplot(111, projection='3d')

    # Plot trajectories with correct z-ordering
    ax_3d.plot(mx_analytical, my_analytical, mz_analytical, color='lightgrey', 
              linewidth=2, label='Analytical', zorder=1)
    ax_3d.plot(expected_mx, expected_my, expected_mz, color='darkgrey', linewidth=2, 
              linestyle='--', label='RK4', zorder=2)
    ax_3d.plot(mx_np, my_np, mz_np, color='black', linewidth=2, label='Neural IVP', zorder=3)

    # Set axis limits first to ensure proper scaling
    max_range = 1.2
    ax_3d.set_xlim([-max_range, max_range])
    ax_3d.set_ylim([-max_range, max_range])
    ax_3d.set_zlim([-max_range, max_range])

    # Equal aspect ratio
    ax_3d.set_box_aspect([1,1,1])

    # Set viewing angle for better visualisation of all axes
    ax_3d.view_init(elev=25, azim=240)

    # Mark initial and final points
    ax_3d.scatter([mx_np[0]], [my_np[0]], [mz_np[0]], color='black', s=80, 
                 marker='o', zorder=10, edgecolors='black', linewidth=1)
    ax_3d.scatter([mx_np[-1]], [my_np[-1]], [mz_np[-1]], color='black', s=80, 
                 marker='s', zorder=10, edgecolors='black', linewidth=1, facecolors='black')

    # Add text annotations with white background
    ax_3d.text(mx_np[0], my_np[0], mz_np[0], '  Start', fontsize=9, 
              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.8))
    ax_3d.text(mx_np[-1], my_np[-1], mz_np[-1], '  End', fontsize=9,
              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.8))

    # Style the axes
    ax_3d.set_xlabel(r'$m_x$', fontsize=12, labelpad=8)
    ax_3d.set_ylabel(r'$m_y$', fontsize=12, labelpad=8)
    ax_3d.set_zlabel(r'$m_z$', fontsize=12, labelpad=8, rotation=180)

    plt.tight_layout()
    phase_path = os.path.join(paths["figures"], "phase_space_3d.pdf")
    fig_3d.savefig(phase_path, dpi=150, bbox_inches='tight')
    plt.close(fig_3d)

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

    loss_path = os.path.join(paths["figures"], "loss_history.pdf")
    fig_loss.savefig(loss_path, dpi=150, bbox_inches='tight')
    plt.close(fig_loss)

    print("**Saved**:")
    print(f"- Model mx: {mx_path}")
    print(f"- Model my: {my_path}")
    print(f"- Model mz: {mz_path}")
    print(f"- Loss history: {os.path.join(paths['paths'], 'loss_history.json')}")
    print(f"- Trajectories: {traj_path}")
    print(f"- 3D Phase space: {phase_path}")
    print(f"- Loss plot: {loss_path}")
    return


if __name__ == "__main__":
    app.run()
