import sys
from pathlib import Path

import pandas as pd
from TPTBox import to_nii

out = str(Path(__file__).parent.parent.parent)
sys.path.append(out)

from constants import default_dict, out_voting, path_train_poi, target_path
from experiments.generate_refrence.reg import (
    aggregate,
    compute_pvalues,
    compute_run_times,
    evaluate_all_experiments,
    fetch,
    logger,
    plot_score,
    run_all_to_atlas,
    to_mrk,
)


def change_to_label(change: dict):
    if not change:
        return "baseline"
    return ",".join(f"{k}={v}" for k, v in change.items())


if __name__ == "__main__":
    out_mrk = path_train_poi.parent / "mrk"
    to_mrk(path_train_poi, out_mrk)

    target = target_path
    fetch(target, True)

    changes = [
        {},
        ###########
        {"lr": 0.1},
        {"lr": 0.01},
        # {"lr": 0.001},
        {"lr": 0.0001},
        {"lr": 0.00001},
        ###########
        {"min_delta": 0.01},
        {"min_delta": 0.001},
        {"min_delta": 0.0001},
        # {"min_delta": 0.000001},
        {"min_delta": 0.0000001},
        {"min_delta": 0.00000001},
        {"min_delta": 0.000000001},
        ###########
        # {"pyramid_levels": 4, "coarsest_level": 3},
        {"pyramid_levels": 5, "coarsest_level": 4},
        {"pyramid_levels": 3, "coarsest_level": 2},
        ##########
        {"be": 0.0},
        {"be": 0.001},
        # {"be": 0.00001},
        {"be": 0.0000001},
        {"be": 0.000000001},
        {"be": 0.1},
        ##########
        # {"mse": 1},
        {"mse": 0},
        {"mse": 10},
        {"mse": 0.1},
        #########
        {"dice": 0.1},
        {"dice": 0},
        {"dice": 0.001},
        #########
        {"com": 0},
        {"com": 0.1},
        {"com": 0.0000000001},
        #########
        {"dice": 0, "com": 0},
        ########
        # {"transform_name": "FreeFormDeformation"}, # Not invertible
        {"transform_name": "StationaryVelocityFieldTransform"},
        {"transform_name": "DisplacementFieldTransform"},
    ]
    time = {}
    for c in changes:
        di = default_dict.copy()
        for k, v in c.items():
            di[k] = v
        # key = "_".join(f"{k}-{v}" for k, v in di.items() if v != "SVFFD")
        key = "_".join(f"{k}-{v}" for k, v in c.items() if v != "SVFFD")
        target = to_nii(target, True)
        out_folder = path_train_poi.parent / f"treg_{key}"
        out_folder.mkdir(exist_ok=True)
        try:
            out_file = run_all_to_atlas(out_mrk, target, out_folder, **di, no_inference=True)
            if out_file is None:
                continue
            stats = compute_run_times(out_file, drop_n=3)
            # print(str(c))
            # print(f"Average runtime (trimmed): {stats['mean']:.2f} s")
            # print(f"Median runtime: {stats['median']:.2f} s")
            # print(f"Std runtime: {stats['std']:.2f} s")
            del stats["all_durations"]
            del stats["used_durations"]
            time["treg_" + key] = stats
            aggregate(out_folder)
        except NotImplementedError:
            logger.print_error()
            break
        except Exception:
            logger.print_error()
            continue
        # break

    df_global, per_lid = evaluate_all_experiments(out_voting)
    # TODO replace the name with only what changed in the dict. Remove _ and -
    baseline = df_global.index[0]  # or explicit name

    df_p = compute_pvalues(per_lid, baseline, score="mean_dist")
    df_final = df_global.join(df_p[["p_value"]])

    ###
    rename_map = {}
    df_index = list(df_final.index)

    for i, change in enumerate(changes):
        di = default_dict.copy()
        for k, v in change.items():
            di[k] = v
        key = "_".join(f"{k}-{v}" for k, v in di.items() if v != "SVFFD")

        old_name = "treg_" + key
        # assert old_name in df_index, (old_name, df_index)
        new_name = change_to_label(change)
        rename_map[old_name] = new_name
    # rename_map["treg"] = "baseline-"
    # rename_map["treg10"] = "min_delta=0.000001"
    # rename_map["treg100"] = "min_delta=0.0000001"
    # rename_map["treg1000"] = "min_delta=0.00000001"
    df_time = pd.DataFrame.from_dict(time, orient="index")
    print(df_time)
    # Now join with df_final safely
    df_final = df_final.join(df_time, how="left")
    # rename_map["baseline"] = "treg"
    # rename_map["baseline"] = "treg"
    df_final = df_final.rename(index=rename_map)
    baseline = rename_map.get(baseline, baseline)
    ###

    print(df_final.round(5).sort_values("mean_dist"))
    df_final.to_excel(out_voting / "summery.xlsx")
    for score in ["mean_dist", "median_dist", "p95_dist"]:
        plot_score(df_final.sort_values("mean_dist"), score, baseline)
