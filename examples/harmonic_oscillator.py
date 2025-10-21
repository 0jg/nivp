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
        gradient,
        expected_path_tensor,
        ensure_run_dirs,
    )
    from systems import HarmonicOscillator
    return (
        CoupledOptimiser,
        DATA_TYPE,
        HarmonicOscillator,
        SimpleMLP,
        ensure_run_dirs,
        expected_path_tensor,
        gradient,
        json,
        os,
    )


@app.cell
def controls():
    # Harmonic oscillator: d^2x/dt^2 = -x
    # With x(0) = 0, v(0) = 1, analytical solution is x(t) = sin(t)
    # Energy = 1/2 * (v^2 + x^2) = 1/2 (conserved)
    params = {
        "activation": "sechlu",
        "hidden": 64,
        "epochs": 2_000,
        "batch_size": 32,
        "lr": 3e-3,
        "T_MIN": 0.0,
        "T_MAX": 10.0,
        "TIMESTEP": 0.05,
        "x0": 0.0,
        "v0": 1.0,
        "ODE_REGULARISER": 1.0,
        "POSITION_REGULARISER": 1.0,
        "VELOCITY_REGULARISER": 1.0,
        "JOB_ID": 1,
    }
    return (params,)


@app.cell
def dataset(
    DATA_TYPE,
    HarmonicOscillator,
    expected_path_tensor,
    params,
    torch,
):
    T_MIN, T_MAX, DT = params["T_MIN"], params["T_MAX"], params["TIMESTEP"]
    grid = torch.linspace(T_MIN, T_MAX, int(round((T_MAX-T_MIN)/DT))+1, dtype=DATA_TYPE)
    t = grid.unsqueeze(0).unsqueeze(2).requires_grad_(True)
    t_dataset = torch.utils.data.TensorDataset(t)
    domain = torch.utils.data.DataLoader(dataset=t_dataset, batch_size=params["batch_size"], shuffle=True)

    x0 = torch.tensor([params["x0"]], dtype=DATA_TYPE)
    v0 = torch.tensor([params["v0"]], dtype=DATA_TYPE)

    system = HarmonicOscillator()
    expected = expected_path_tensor(system, x0, v0, T_MIN, T_MAX, DT)
    return domain, expected, t


@app.cell
def train_loop(
    CoupledOptimiser,
    DATA_TYPE,
    SimpleMLP,
    domain,
    ensure_run_dirs,
    expected,
    gradient,
    json,
    os,
    params,
    plt,
    t,
    torch,
    tqdm,
):
    # Build model (single network for 1D position)
    model_x = SimpleMLP(in_features=1, hidden=params["hidden"], out_features=1, activation=params["activation"])
    opt_x = torch.optim.Adam(model_x.parameters(), lr=params["lr"])
    optimiser = CoupledOptimiser(opt_x)

    def train():
        EPOCHS = params["epochs"]
        loss_hist = []
        for epoch in tqdm(range(EPOCHS)):
            domain_list = list(domain)
            for t_batch_tuple in domain_list:
                t_batch = t_batch_tuple[0]
                px = model_x(t_batch).requires_grad_(True)

                vx = gradient(px, t_batch, 1)

                # ODE residual: d²x/dt² + x = 0
                ode_x = gradient(px, t_batch, 2) + px

                l2 = torch.nn.MSELoss()
                pos_x = l2(torch.tensor(params["x0"], dtype=DATA_TYPE).unsqueeze(0), px[:,0,:])
                vel_x = l2(torch.tensor(params["v0"], dtype=DATA_TYPE).unsqueeze(0).unsqueeze(0), vx[0][0].unsqueeze(0))

                ode_loss_x = l2(ode_x, torch.zeros_like(px))

                overall_x = params["ODE_REGULARISER"]*ode_loss_x + params["POSITION_REGULARISER"]*pos_x + params["VELOCITY_REGULARISER"]*vel_x
                overall = overall_x
                loss_hist.append(float(overall.detach().cpu().numpy()))

                optimiser.zero_grad()
                overall.backward(retain_graph=True)
                optimiser.step()

            if (epoch % 500 == 0) or (epoch == EPOCHS-1):
                fig = plt.figure(figsize=(10, 6), dpi=150)
                fig.patch.set_facecolor('white')

                tt = t[0,:,0].detach().cpu().numpy()
                px_np = px.detach().cpu().numpy()[0,:,0]
                expected_x = expected.detach().cpu().numpy()[0,0,0,:]

                # Create plot
                ax = fig.add_subplot(111)
                ax.plot(tt, px_np, color='black', linewidth=2, label='Neural IVP')
                ax.plot(tt, expected_x, color='lightgrey', linewidth=2.5, label='RK4', zorder=-1)
                ax.set_xlabel(r'$t$', fontsize=12)
                ax.set_ylabel(r'$x(t)$', fontsize=12)
                ax.legend(loc='upper right', frameon=False, fontsize=11)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.tick_params(direction='in', which='both', left=True, bottom=True)
                ax.tick_params(labelbottom=True, labelleft=True)

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

        return model_x, loss_hist, px

    model_x, loss_hist, px_final = train()

    # Save results to examples/outputs
    output_base = os.path.join(os.path.dirname(__file__), "outputs")
    paths = ensure_run_dirs(output_base, params, system_name="harmonic_oscillator")

    # Save model
    mx_path = os.path.join(paths["models"], f'{params["JOB_ID"]}_x.pth')
    torch.save(model_x, mx_path)

    # Save loss history
    with open(os.path.join(paths["paths"], "loss_history.json"), "w") as f:
        json.dump(loss_hist, f)

    # Generate and save final figures
    tt = t[0,:,0].detach().cpu().numpy()
    px_np = px_final.detach().cpu().numpy()[0,:,0]
    expected_x = expected.detach().cpu().numpy()[0,0,0,:]

    # Save trajectory figure
    fig = plt.figure(figsize=(10, 6), dpi=150)
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111)
    ax.plot(tt, px_np, color='black', linewidth=2, label='Neural IVP')
    ax.plot(tt, expected_x, color='lightgrey', linewidth=2.5, label='RK4', zorder=-1)
    ax.set_xlabel(r'$t$', fontsize=12)
    ax.set_ylabel(r'$x(t)$', fontsize=12)
    ax.legend(loc='upper right', frameon=False, fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='in', which='both', left=True, bottom=True)
    ax.tick_params(labelbottom=True, labelleft=True)

    traj_path = os.path.join(paths["figures"], "trajectory.png")
    fig.savefig(traj_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

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
    print(f"- Model: {mx_path}")
    print(f"- Loss history: {os.path.join(paths['paths'], 'loss_history.json')}")
    print(f"- Trajectory: {traj_path}")
    print(f"- Loss plot: {loss_path}")
    return


if __name__ == "__main__":
    app.run()
