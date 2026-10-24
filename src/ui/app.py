import gradio as gr

from pl_model.model import predict_match
from pl_model.simulate import simulate_seasons

TABLE_COLUMNS = {
    "AvgPoints": "Points",
    "AvgWon": "W",
    "AvgDrawn": "D",
    "AvgLost": "L",
    "AvgGoalsScored": "GF",
    "AvgGoalsConceded": "GA",
    "TitlePct": "Title %",
    "Top4Pct": "Top 4 %",
    "RelegationPct": "Relegated %",
}


def build_ui(params, fixtures, max_n_simulations):
    def predict(home_team, away_team):
        if home_team == away_team:
            raise gr.Error("Home and away teams must be different.")

        result = predict_match(params, home_team, away_team, n_scorelines=5)

        outcomes = {
            f"{home_team} win": result["home_win"],
            "Draw": result["draw"],
            f"{away_team} win": result["away_win"],
        }
        expected_goals = (
            f"**Expected goals:** {home_team} {result['expected_home_goals']:.2f} "
            f"- {result['expected_away_goals']:.2f} {away_team}"
        )
        scorelines = {
            f"{s['home_goals']}-{s['away_goals']}": s["probability"]
            for s in result["most_likely_scorelines"]
        }
        return outcomes, expected_goals, scorelines

    def simulate(n_simulations, seed):
        table = simulate_seasons(params, fixtures, n_simulations=int(n_simulations), seed=int(seed))
        table = table.rename(columns=TABLE_COLUMNS).round(1).reset_index()
        table.insert(0, "Pos", range(1, len(table) + 1))
        return table[["Pos", "Team", *TABLE_COLUMNS.values()]]

    current_teams = sorted({team for fixture in fixtures for team in fixture})

    with gr.Blocks(title="Footy Forecast") as demo:
        gr.Markdown("# Footy Forecast\nDixon-Coles model fitted on past Premier League results.")

        with gr.Tab("Match"):
            with gr.Row():
                home_team = gr.Dropdown(choices=current_teams, value=current_teams[0], label="Home team")
                away_team = gr.Dropdown(choices=current_teams, value=current_teams[1], label="Away team")
            predict_button = gr.Button("Predict", variant="primary")

            with gr.Row():
                outcomes = gr.Label(label="Result")
                scorelines = gr.Label(label="Most likely scorelines")
            expected_goals = gr.Markdown()

            predict_button.click(
                predict,
                inputs=[home_team, away_team],
                outputs=[outcomes, expected_goals, scorelines],
            )

        with gr.Tab("Season"):
            gr.Markdown("Simulates the full 2026/27 fixture list many times and averages the final tables.")
            with gr.Row():
                n_simulations = gr.Slider(
                    minimum=100, maximum=max_n_simulations, step=100, value=1000, label="Simulations"
                )
                seed = gr.Number(value=42, precision=0, label="Seed")
            simulate_button = gr.Button("Simulate season", variant="primary")
            table = gr.Dataframe(label="Projected table", interactive=False)

            simulate_button.click(simulate, inputs=[n_simulations, seed], outputs=table)

    return demo
