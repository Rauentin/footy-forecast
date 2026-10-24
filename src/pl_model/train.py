import time

from pl_model.config import Settings
from pl_model.model import fit, load_data


def main():
    settings = Settings()

    df = load_data(settings.data_path)
    print(f"Fitting on {len(df)} matches from {settings.data_path} (xi={settings.xi}) ...")

    start = time.perf_counter()
    params = fit(df, xi=settings.xi)
    params.save(settings.params_path)

    print(f"Fit converged: {params.info['converged']} in {time.perf_counter() - start:.0f}s")
    print(f"Log-likelihood: {params.info['log_likelihood']:.2f}")
    print(f"Home advantage: {params.home_adv:.4f}")
    print(f"Rho: {params.rho:.4f}")
    print(f"Saved to {settings.params_path}")


if __name__ == "__main__":
    main()
