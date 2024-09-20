import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shap
from xgboost import *

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_curve,
    auc,
    average_precision_score,
    precision_recall_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    train_test_split,
    StratifiedKFold,
)
from sklearn.preprocessing import LabelEncoder

from datetime import date
import time


np.random.seed(1)


class CellPhenoX:
    def __init__(self, X, y, CV_repeats, outer_num_splits, inner_num_splits):
        """_summary_

        Args:
            X (dataframe): cell by latent dimensions dataframe
            y (series): the target variable
            CV_repeats (int): number of times to repeat the cross-validation
            outer_num_splits (int): number of outer folds (stratified k fold)
            inner_num_splits (int): number of inner folds (for hyperparameter tuning)
        """
        # self.steps = ["hyperparameter tuning", "model training", "model prediction", "performance eval", "SHAP value calculation"]
        # self.CV_repeat_cumulative_times = [0,0,0,0,0]
        self.CV_repeat_times = []
        self.X = X
        self.y = y
        # self.fc = fc
        # self.num_samp = num_samp
        self.CV_repeats = CV_repeats
        self.outer_num_splits = outer_num_splits
        self.inner_num_splits = inner_num_splits
        # Make a list of random integers between 0 and 10000 of length = CV_repeats to act as different data splits
        self.random_states = np.random.randint(10000, size=CV_repeats)
        self.param_grid = [
            {
                "max_features": ["sqrt", "log2"],
                "max_depth": [10, 20, 30],
                "min_samples_leaf": [1, 2, 5],
                "min_samples_split": [2, 5, 10],
                "n_estimators": [100, 200, 800],
            }
        ]
        self.label_encoder = LabelEncoder()
        self.best_model = None
        self.best_score = float("-inf")
        self.shap_values_per_cv = dict()
        self.shap_df = None
        for sample in X.index:
            ## Create keys for each sample
            self.shap_values_per_cv[sample] = {}
            ## Then, keys for each CV fold within each sample
            for CV_repeat in range(self.CV_repeats):
                self.shap_values_per_cv[sample][CV_repeat] = {}

    def split_data(self, train_outer_ix, test_outer_ix):
        X_train_outer, X_test_outer = (
            self.X.iloc[train_outer_ix, :],
            self.X.iloc[test_outer_ix, :],
        )
        y_train_outer, y_test_outer = (
            self.y.iloc[train_outer_ix],
            self.y.iloc[test_outer_ix],
        )

        # Create an additional inner split for validation
        X_train_inner, X_val_inner, y_train_inner, y_val_inner = train_test_split(
            X_train_outer,
            y_train_outer,
            test_size=0.2,
            random_state=42,
            stratify=y_train_outer,
        )
        y_train_inner = self.label_encoder.fit_transform(y_train_inner)
        y_train_outer = self.label_encoder.fit_transform(y_train_outer)
        y_test_outer = self.label_encoder.transform(y_test_outer)
        y_val_inner = self.label_encoder.transform(y_val_inner)

        return [
            X_train_outer,
            X_test_outer,
            X_train_inner,
            X_val_inner,
            y_train_inner,
            y_train_outer,
            y_test_outer,
            y_val_inner,
        ]

    def model_training_shap_val(self, outpath):
        """Train the model using nested cross validation strategy and generate shap values for each fold/CV repeat

        Parameters:
        outpath (str): the path for the output folder

        Returns:


        """

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
        accuracy_list, auc_list = [], []
        y_test_combined_list, y_prob_combined_list = [], []
        val_auc_combined_list, val_accuracy_combined_list, val_prc_combined_list = (
            [],
            [],
            [],
        )

        # save the models
        overal_model_list = []
        best_model = None

        print("entering CV loop")
        for i, CV_repeat in enumerate(range(self.CV_repeats)):
            # Verbose
            print("\n------------ CV Repeat number:", CV_repeat + 1)
            # Establish CV scheme
            CV = StratifiedKFold(
                n_splits=self.outer_num_splits,
                shuffle=True,
                random_state=self.random_states[i],
            )

            ix_training, ix_test = [], []
            # Loop through each fold and append the training & test indices to the empty lists above
            for fold in CV.split(self.X, self.y):
                ix_training.append(fold[0]), ix_test.append(fold[1])

            y_prob_list = []
            y_test_list = []
            val_auc_list = []
            val_accuracy_list = []
            val_prc_list = []
            model_list = []

            # hp, mt, mp, pe, sv = [],[],[],[],[]

            ## Loop through each outer fold and extract SHAP values
            for j, (train_outer_ix, test_outer_ix) in enumerate(
                zip(ix_training, ix_test)
            ):
                # Verbose
                print("\n------ Fold Number:", j + 1)
                (
                    X_train_outer,
                    X_test_outer,
                    X_train_inner,
                    X_val_inner,
                    y_train_inner,
                    y_train_outer,
                    y_test_outer,
                    y_val_inner,
                ) = self.split_data(train_outer_ix, test_outer_ix)
                ## Establish inner CV for parameter optimization #-#-#
                cv_inner = StratifiedKFold(
                    n_splits=self.inner_num_splits, shuffle=True, random_state=1
                )  # -#-#

                # Search to optimize hyperparameters
                # hp_start = time.time()
                model = RandomForestClassifier(random_state=10, class_weight="balanced")
                search = RandomizedSearchCV(
                    model,
                    self.param_grid,
                    scoring="balanced_accuracy",
                    cv=cv_inner,
                    refit=True,
                    n_jobs=1,
                )  # -#-#
                result = search.fit(X_train_inner, y_train_inner)  # -#=#
                # hp_end = time.time()
                # hp.append(hp_end - hp_start)
                # self.CV_repeat_cumulative_times[0]+= (hp_end - hp_start)

                # Fit model on training data
                # mt_start = time.time()
                result.best_estimator_.fit(X_train_outer, y_train_outer)  # -#-#
                # mt_end = time.time()
                # mt.append(mt_end - mt_start)
                # self.CV_repeat_cumulative_times[1] += (mt_end - mt_start)

                # Make predictions on the test set
                # mp_start = time.time()
                y_pred = result.best_estimator_.predict(X_test_outer)
                y_prob = result.best_estimator_.predict_proba(X_test_outer)[:, 1]
                y_prob_list.append(y_prob)
                y_test_list.append(y_test_outer)
                # mp_end = time.time()
                # mp.append(mp_end - mp_start)
                # self.CV_repeat_cumulative_times[2] +=(mp_end - mp_start)

                # Calculate accuracy
                # pe_start = time.time()
                accuracy = accuracy_score(y_test_outer, y_pred)
                print("--- Accuracy: ", accuracy)
                accuracy_list.append(accuracy)

                # Calculate ROC curve and AUC
                fpr, tpr, thresholds = roc_curve(y_test_outer, y_pred)
                auc_value = auc(fpr, tpr)
                auc_list.append(auc_value)
                # Validate on the validation set
                y_val_pred = result.best_estimator_.predict(X_val_inner)
                val_accuracy = accuracy_score(y_val_inner, y_val_pred)
                val_accuracy_list.append(val_accuracy_list)
                fpr_val, tpr_val, _ = roc_curve(y_val_inner, y_val_pred)
                y_prob_val = result.best_estimator_.predict_proba(X_val_inner)[:, 1]
                val_auc = auc(fpr_val, tpr_val)
                val_prc = average_precision_score(y_val_inner, y_prob_val)
                val_auc_list.append(val_auc)
                print(len(val_auc_list))
                val_prc_list.append(val_prc)
                val_accuracy_list.append(val_accuracy)
                print(
                    "--- Validation Accuracy: ",
                    val_accuracy,
                    " - Validation AUROC: ",
                    val_auc,
                    " - Val AUPRC: ",
                    val_prc,
                )
                # pe_end = time.time()
                # pe.append(pe_end - pe_start)
                # self.CV_repeat_cumulative_times[3] += (pe_end - pe_start)

                # TODO: could probably save the explainer for this modeltrain object
                # Use SHAP to explain predictions using best estimator
                # sv_start = time.time()
                explainer = shap.TreeExplainer(result.best_estimator_)
                shap_values = explainer.shap_values(X_test_outer)

                # Extract SHAP information per fold per sample
                for k, test_index in enumerate(test_outer_ix):
                    test_index = self.X.index[test_index]
                    # TODO: here, I am selecting the second (1) shap array for a binary classification problem.
                    # TODO: we need a way to generalize this so that we select the array that corresponds to the
                    # TODO: positive class (disease).
                    self.shap_values_per_cv[test_index][CV_repeat] = shap_values[1][k]
                sv_end = time.time()
                # sv.append(sv_end - sv_start)
                # self.CV_repeat_cumulative_times[4] += (sv_end - sv_start)

                # save best model
                model_list.append(result.best_estimator_)

            # one ROC curve for each repeat
            y_prob_combined = np.concatenate(y_prob_list)
            y_test_combined = np.concatenate(y_test_list)
            y_prob_combined_list.append(y_prob_combined)
            y_test_combined_list.append(y_test_combined)
            val_auc_combined = val_auc_list
            val_auc_combined_list.append(val_auc_combined)
            val_accuracy_combined = val_accuracy_list
            val_accuracy_combined_list.append(val_accuracy_combined)
            val_prc_combined = val_prc_list
            val_prc_combined_list.append(val_prc_combined)

            # Compute ROC and PR curve for the combined data
            fpr, tpr, _ = roc_curve(y_test_combined, y_prob_combined)
            precision, recall, thresholds = precision_recall_curve(
                y_test_combined, y_prob_combined
            )
            # Compute AUC (Area Under the Curve)
            roc_auc = auc(fpr, tpr)
            prc_auc = average_precision_score(y_test_combined, y_prob_combined)

            # save the best model for this repeat
            model_pr_pairs = list(zip(model_list, val_prc_combined))

            # Find the model with the highest precision-recall score
            best_model_repeat, best_score_repeat = max(
                model_pr_pairs, key=lambda x: x[1]
            )
            overal_model_list.append((best_model_repeat, best_score_repeat))

            # Plot the ROC and precision recall curve for the current fold size
            axes[0].plot(fpr, tpr, lw=2, label=f"CV Repeat {i+1} (ROC = {roc_auc:.2f})")
            axes[1].plot(
                recall, precision, lw=2, label=f"CV Repeat{i+1} (PRC = {prc_auc:.2f})"
            )

            # self.CV_repeat_times.append([mean(hp), mean(mt), mean(mp), mean(pe), mean(sv)])

        # Add labels and show the ROC curve plot
        today = date.today()
        d = today.strftime("%m%d%y")

        axes[0].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        axes[0].set_xlim([0.0, 1.0])
        axes[0].set_ylim([0.0, 1.05])
        axes[0].set_xlabel("False Positive Rate")
        axes[0].set_ylabel("True Positive Rate")
        axes[0].set_title(
            "Receiver Operating Characteristic\naggregated over folds for each repeat"
        )
        axes[0].legend(loc="lower right")

        axes[1].set_xlim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.05])
        axes[1].set_xlabel("False Positive Rate")
        axes[1].set_ylabel("True Positive Rate")
        axes[1].set_title(
            "Precision Recall Curve\naggregated over folds for each repeat"
        )
        axes[1].legend(loc="lower right")

        y_prob_combined_repeat = np.concatenate(y_prob_combined_list)
        y_test_combined_repeat = np.concatenate(y_test_combined_list)

        predicted_positive = y_prob_combined_repeat[y_test_combined_repeat == 1]
        predicted_negative = y_prob_combined_repeat[y_test_combined_repeat == 0]
        axes[2].hist(
            predicted_negative,
            label="Negative Class",
            bins=30,
            color="#f09102",
            alpha=0.5,
        )
        axes[2].hist(
            predicted_positive,
            label="Positive Class",
            bins=30,
            color="#2d5269",
            alpha=0.5,
        )
        axes[2].set_xlabel("Predicted Probabilities")
        axes[2].set_ylabel("Frequency")
        axes[2].set_title("Distribution of Predicted Probabilities")
        axes[2].legend(loc="upper right", labels=["Negative Class", "Positive Class"])
        plt.suptitle("Classification Model Performance Evaluation")
        plt.tight_layout()
        plt.savefig(f"{outpath}modelperformance.pdf", format="pdf")

        val_auc_combined_repeat = np.concatenate(val_auc_combined_list)
        val_prc_combined_repeat = np.concatenate(val_prc_combined_list)

        avg_val_auc = np.mean(val_auc_combined_repeat)
        avg_val_prc = np.mean(val_prc_combined_repeat)
        print(f"Average AUROC: {avg_val_auc} | Average AUPRC: {avg_val_prc}")

        # select the final model
        self.best_model, self.best_score = max(overal_model_list, key=lambda x: x[1])
        print(f"best model precision-recall score = {self.best_score:.4f}")

        # self.plot_time(outpath)

        # now aggregate the shap values per CV
        self.get_shap_values(outpath)
        # and calculate the interpretable score
        self.get_interpretable_score()

    def get_shap_values_per_cv(self):
        return self.shap_values_per_cv

    def get_best_score(self):
        return self.best_score

    def get_best_model(self):
        return self.best_model

    def get_shap_values(self, outpath):
        # Establish lists to keep average Shap values, their Stds, and their min and max
        average_shap_values = []
        for i in range(0, len(self.X)):  # len(NAM)
            id = self.X.index[i]  # NAM.index[i]
            # print(id)
            df_per_obs = pd.DataFrame.from_dict(
                self.shap_values_per_cv[id]
            )  # Get all SHAP values for sample number i
            # Get relevant statistics for every sample
            average_shap_values.append(df_per_obs.mean(axis=1).values)
        self.shap_df = pd.DataFrame(
            average_shap_values, columns=[f"{col}_shap" for col in self.X.columns]
        )
        self.shap_df = self.shap_df.set_index(self.X.index)
        # plot the SHAP summary plot?
        plt.figure()
        shap.summary_plot(np.array(average_shap_values), self.X, show=False)
        plt.title("Average SHAP values after nested cross-validation")
        plt.savefig(f"{outpath}SHAPsummary.png")

    def get_interpretable_score(self):
        # Calculate the SHAP-adjusted probability score
        # y_prob_all = self.best_model.predict_proba(self.X)[:, 1] # no longer needed if we are not scaling the prediceted probabilities
        interpretable_score = np.sum(self.shap_df, axis=1)
        # add the shap_df
        self.shap_df["interpretable_score"] = interpretable_score

    """def plot_time(self, outpath):
        #plt.figure(figsize=(10,6))
        # for cv in range(len(self.CV_repeat_times)):
        #     plt.plot(self.steps, self.CV_repeat_times[cv], marker="o", label=f"CV repeat {cv+1}")
        # plt.xlabel("Steps")
        # plt.ylabel("Time (seconds)")
        # plt.title("Time for each step\naveraged across folds for each repeat\n(#samples=30, fc=3)")
        # plt.grid(True)
        # plt.savefig(f"{outpath}sim_benchtime_numsamp30_fc3.png")
        time_cumu = np.cumsum(self.CV_repeat_cumulative_times)
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
        for cv in range(len(self.CV_repeat_times)):
            axes[0].plot(self.steps, self.CV_repeat_times[cv], marker="o", label=f"CV repeat {cv+1}")
        axes[1].plot(self.steps, time_cumu, marker="o")
            
        axes[0].set_xlabel("Steps")
        axes[0].set_ylabel("Time (seconds)")
        axes[0].set_title("Time for each step\naveraged across folds for each repeat\n(#samples=30, fc=3)")
        axes[0].legend()
        axes[0].grid(True)
        axes[1].set_xlabel("Steps")
        axes[1].set_ylabel("Cumulative time (seconds)")
        axes[1].set_title("Cumulative time for each step\nacross folds and repeats")
        for tick in axes[0].get_xticklabels():
            tick.set_rotation(45)
        for tick in axes[1].get_xticklabels():
            tick.set_rotation(45)

        axes[1].grid(True)
        plt.tight_layout()
        plt.savefig(f"{outpath}sim_benchtime_numsamp30_fc3_ext_allCells.pdf", format="pdf")"""
