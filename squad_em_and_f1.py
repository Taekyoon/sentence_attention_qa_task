import squad_eval


class SquadEmAndF1:
    """
    This :class:`Metric` takes the best span string computed by a model, along with the answer
    strings labeled in the data, and computed exact match and F1 score using the official SQuAD
    evaluation script.
    """
    def __init__(self):
        self._total_em = 0.0
        self._total_f1 = 0.0
        self._count = 0

    def __call__(self, best_span_string, answer_strings):
        """
        Parameters
        ----------
        value : ``float``
            The value to average.
        """
        exact_match = squad_eval.metric_max_over_ground_truths(
                squad_eval.exact_match_score,
                best_span_string,
                answer_strings)
        f1_score = squad_eval.metric_max_over_ground_truths(
                squad_eval.f1_score,
                best_span_string,
                answer_strings)
        self._total_em += exact_match
        self._total_f1 += f1_score
        self._count += 1

        return exact_match

    def get_metric(self, reset=False):
        """
        Returns
        -------
        Average exact match and F1 score (in that order) as computed by the official SQuAD script
        over all inputs.
        """
        exact_match = self._total_em / self._count if self._count > 0 else 0
        f1_score = self._total_f1 / self._count if self._count > 0 else 0
        if reset:
            self.reset()
        return exact_match, f1_score

    def reset(self):
        self._total_em = 0.0
        self._total_f1 = 0.0
        self._count = 0
