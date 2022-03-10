from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .bilateral import Bilateral
from .utils import HDFMixin, fast_binomial_pmf


class MidlineBilateral(HDFMixin):
    """Model a bilateral lymphatic system where an additional risk factor can
    be provided in the data: Whether or not the primary tumor extended over the
    mid-sagittal line.

    It is reasonable to assume (and supported by data) that such an extension
    significantly increases the risk for metastatic spread to the contralateral
    side of the neck. This class attempts to capture this using a simple
    assumption: We assume that the probability of spread to the contralateral
    side for patients *with* midline extension is larger than for patients
    *without* it, but smaller than the probability of spread to the ipsilateral
    side. Formally:

    .. math::
        b_c^{\\in} = \\alpha \\cdot b_i + (1 - \\alpha) \\cdot b_c^{\\not\\in}

    where :math:`b_c^{\\in}` is the probability of spread from the primary tumor
    to the contralateral side for patients with midline extension, and
    :math:`b_c^{\\not\\in}` for patients without. :math:`\\alpha` is the linear
    mixing parameter.
    """

    def __init__(
        self,
        graph: Dict[Tuple[str], List[str]] = {},
        alpha_mix: float = 0.,
        trans_symmetric: bool = True,
        **kwargs
    ):
        """The class is constructed in a similar fashion to the
        :class:`Bilateral`: That class contains one :class:`Unilateral` for
        each side of the neck, while this class will contain two instances of
        :class:`Bilateral`, one for the case of a midline extension and one for
        the case of no midline extension.

        Args:
            graph: Dictionary of the same kind as for initialization of
                :class:`System`. This graph will be passed to the constructors of
                two :class:`System` attributes of this class.
            alpha_mix: Initial mixing parameter between ipsi- & contralateral
                base probabilities that determines the contralateral base
                probabilities for the patients with mid-sagittal extension.
            trans_symmetric: If ``True``, the spread probabilities among the
                LNLs will be set symmetrically.

        See Also:
            :class:`Bilateral`: Two of these are held as attributes by this
            class. One for the case of a mid-sagittal extension of the primary
            tumor and one for the case of no such extension.
        """
        self.ext   = Bilateral(
            graph=graph, base_symmetric=False, trans_symmetric=trans_symmetric
        )
        self.noext = Bilateral(
            graph=graph, base_symmetric=False, trans_symmetric=trans_symmetric
        )
        self.alpha_mix = alpha_mix


    @property
    def graph(self) -> Dict[Tuple[str], List[str]]:
        """Return the (unilateral) graph that was used to create this network.
        """
        return self.noext.graph


    @property
    def base_probs(self) -> np.ndarray:
        """Base probabilities of metastatic lymphatic spread from the tumor(s)
        to the lymph node levels. This will return a concatenation of the
        ipsilateral base probabilities and the contralateral ones without the
        midline extension, as well as - lastly - the mixing parameter alpha.
        The returned array has therefore this composition:

        +-----------------+-------------------+--------------+
        | base probs ipsi | base probs contra | mixing param |
        +-----------------+-------------------+--------------+

        When setting these, one also needs to provide this mixing parameter as
        the last entry in the provided array.
        """
        return np.concatenate([self.noext.base_probs, [self.alpha_mix]])

    @base_probs.setter
    def base_probs(self, new_params: np.ndarray):
        """Set the base probabilities from the tumor(s) to the LNLs, accounting
        for the mixing parameter :math:`\\alpha``.
        """
        new_base_probs = new_params[:-1]
        self.alpha_mix = new_params[-1]

        # base probabilities for lateralized cases
        self.noext.base_probs = new_base_probs

        # base probabilities for cases with tumors extending over the midline
        self.ext.ipsi.base_probs = self.noext.ipsi.base_probs
        self.ext.contra.base_probs = (
            self.alpha_mix * self.noext.ipsi.base_probs
            + (1 - self.alpha_mix) * self.noext.contra.base_probs
        )

        # avoid unnecessary double computation of ipsilateral transition matrix
        self.noext.ipsi._transition_matrix = self.ext.ipsi.transition_matrix


    @property
    def trans_probs(self) -> np.ndarray:
        """Probabilities of lymphatic spread among the lymph node levels. They
        are assumed to be symmetric ipsi- & contralaterally by default.
        """
        return self.noext.trans_probs

    @trans_probs.setter
    def trans_probs(self, new_params: np.ndarray):
        """Set the new spread probabilities for lymphatic spread from among the
        LNLs.
        """
        self.noext.trans_probs = new_params
        self.ext.trans_probs = new_params

        # avoid unnecessary double computation of ipsilateral transition matrix
        self.noext.ipsi._transition_matrix = self.ext.ipsi.transition_matrix


    @property
    def spread_probs(self) -> np.ndarray:
        """These are the probabilities representing the spread of cancer along
        lymphatic drainage pathways per timestep.

        The returned array here contains the probabilities of spread from the
        tumor(s) to the ipsilateral LNLs, then the same values for the spread
        to the contralateral LNLs, after this the spread probabilities among
        the LNLs (which is assumed to be symmetric ipsi- & contralaterally) and
        finally the mixing parameter :math:`\\alpha`. So, it's form is

        +-----------------+-------------------+-------------+--------------+
        | base probs ipsi | base probs contra | trans probs | mixing param |
        +-----------------+-------------------+-------------+--------------+
        """
        spread_probs = self.noext.spread_probs
        return np.concatenate([spread_probs, [self.alpha_mix]])

    @spread_probs.setter
    def spread_probs(self, new_params: np.ndarray):
        """Set the new spread probabilities and the mixing parameter
        :math:`\\alpha`.
        """
        num_base_probs = len(self.noext.ipsi.base_edges)

        new_base_probs  = new_params[:2*num_base_probs]
        new_trans_probs = new_params[2*num_base_probs:-1]
        alpha_mix = new_params[-1]

        self.base_probs = np.concatenate([new_base_probs, [alpha_mix]])
        self.trans_probs = new_trans_probs


    @property
    def modalities(self):
        """A dictionary containing the specificity :math:`s_P` and sensitivity
        :math:`s_N` values for each diagnostic modality.

        Such a dictionary can also be provided to set this property and compute
        the observation matrices of all used systems.

        See Also:
            :meth:`Bilateral.modalities`: Getting and setting this property in
            the normal bilateral model.

            :meth:`Unilateral.modalities`: Getting and setting :math:`s_P` and
            :math:`s_N` for a unilateral model.
        """
        return self.noext.modalities

    @modalities.setter
    def modalities(self, modality_spsn: Dict[str, List[float]]):
        """Call the respective getter and setter methods of the bilateral
        components with and without midline extension.
        """
        self.noext.modalities = modality_spsn
        self.ext.modalities = modality_spsn


    @property
    def patient_data(self):
        """A pandas :class:`DataFrame` with rows of patients and columns of
        patient and involvement details. The table's header should have three
        levels that categorize the individual lymph node level's involvement to
        the corresponding diagnostic modality (first level), the side of the
        LNL (second level) and finaly the name of the LNL (third level).
        Additionally, the patient's T-category must be stored under ('info',
        'tumor', 't_stage') and whether the tumor extends over the mid-sagittal
        line should be noted under ('info', 'tumor', 'midline_extension'). So,
        part of this table could look like this:

        +-----------------------------+------------------+------------------+
        |            info             |       MRI        |       PET        |
        +-----------------------------+--------+---------+--------+---------+
        |            tumor            |  ipsi  | contra  |  ipsi  | contra  |
        +---------+-------------------+--------+---------+--------+---------+
        | t_stage | midline_extension |   II   |   II    |   II   |   II    |
        +=========+===================+========+=========+========+=========+
        | early   | ``True``          |``True``|``None`` |``True``|``False``|
        +---------+-------------------+--------+---------+--------+---------+
        | late    | ``True``          |``None``|``None`` |``None``|``None`` |
        +---------+-------------------+--------+---------+--------+---------+
        | early   | ``False``         |``True``|``False``|``True``|``True`` |
        +---------+-------------------+--------+---------+--------+---------+
        """
        try:
            return self._patient_data
        except AttributeError:
            raise AttributeError(
                "No patient data has been loaded yet"
            )

    @patient_data.setter
    def patient_data(self, patient_data: pd.DataFrame):
        """Load the patient data. For now, this just calls the :meth:`load_data`
        method, but at a later point, I would like to write a function here
        that generates the pandas :class:`DataFrame` from the internal matrix
        representation of the data.
        """
        self._patient_data = patient_data.copy()
        self.load_data(patient_data)


    def load_data(
        self,
        data: pd.DataFrame,
        t_stages: Optional[List[int]] = None,
        modality_spsn: Optional[Dict[str, List[float]]] = None,
        mode = "HMM"
    ):
        """Load data as table of patients with involvement details and convert
        it into internal representation of a matrix.

        Args:
            data: The table with rows of patients and columns of patient and
                involvement details. The table's header must have three levels
                that categorize the individual lymph node level's involvement
                to the corresponding diagnostic modality (first level), the
                side of the LNL (second level) and finaly the name of the LNL
                (third level). Additionally, the patient's T-category must be
                stored under ('info', 'tumor', 't_stage') and whether the tumor
                extends over the mid-sagittal line should be noted under
                ('info', 'tumor', 'midline_extension'). So, part of this table
                could look like this:

                +-----------------------------+---------------------+
                |            info             |         MRI         |
                +-----------------------------+----------+----------+
                |            tumor            |   ipsi   |  contra  |
                +---------+-------------------+----------+----------+
                | t_stage | midline_extension |    II    |    II    |
                +=========+===================+==========+==========+
                | early   | ``True``          | ``True`` | ``None`` |
                +---------+-------------------+----------+----------+
                | late    | ``True``          | ``None`` | ``None`` |
                +---------+-------------------+----------+----------+
                | early   | ``False``         | ``True`` | ``True`` |
                +---------+-------------------+----------+----------+

            t_stages: List of T-stages that should be included in the learning
                process. If ommitted, the list of T-stages is extracted from
                the :class:`DataFrame`
            modality_spsn: If no diagnostic modalities have been defined yet,
                this must be provided to build the observation matrix.

        See Also:
            :attr:`patient_data`: The attribute for loading and exporting data.

            :meth:`Bilateral.load_data`: Loads data into a bilateral network by
            splitting it into ipsi- & contralateral side and passing each to
            the respective unilateral method (see below).

            :meth:`Unilateral.load_data`: Data loading method of the unilateral
            network.
        """
        ext_data = data.loc[data[("info", "tumor", "midline_extension")]]
        noext_data = data.loc[~data[("info", "tumor", "midline_extension")]]

        self.ext.load_data(
            ext_data,
            t_stages=t_stages,
            modality_spsn=modality_spsn,
            mode=mode
        )
        self.noext.load_data(
            noext_data,
            t_stages=t_stages,
            modality_spsn=modality_spsn,
            mode=mode
        )


    def _are_valid_(self, new_spread_probs: np.ndarray) -> bool:
        """Check that the spread probability (rates) are all within limits.
        """
        if new_spread_probs.shape != self.spread_probs.shape:
            msg = ("Shape of provided spread parameters does not match network")
            raise ValueError(msg)
        if np.any(np.greater(0., new_spread_probs)):
            return False
        if np.any(np.greater(new_spread_probs, 1.)):
            return False

        return True


    def log_likelihood(
        self,
        spread_probs: np.ndarray,
        t_stages: Optional[List[Any]] = None,
        diag_times: Optional[Dict[Any, int]] = None,
        max_t: Optional[int] = 10,
        time_dists: Optional[Dict[Any, np.ndarray]] = None
    ) -> float:
        """Compute log-likelihood of (already stored) data, given the spread
        probabilities and either a discrete diagnose time or a distribution to
        use for marginalization over diagnose times.

        Args:
            spread_probs: Spread probabiltites from the tumor to the LNLs, as
                well as from (already involved) LNLs to downsream LNLs. This
                includes both sides of the neck and also different sets of
                probabilities for patients with tumor that do or do not extend
                over the mid-sagittal line. Also includes the :math:`\\alpha`
                mixing parameter. So, this consists of

                +------------+-------------+--------------+
                | base probs | trans probs | mixing param |
                +------------+-------------+--------------+

            t_stages: List of T-stages that are also used in the data to denote
                how advanced the primary tumor of the patient is. This does not
                need to correspond to the clinical T-stages 'T1', 'T2' and so
                on, but can also be more abstract like 'early', 'late' etc.

            diag_times: For each T-stage, one can specify with what time step
                the likelihood should be computed. If this is set to `None`,
                and a distribution over diagnose times `time_dists` is provided,
                the function marginalizes over diagnose times.

            max_t: Latest possible diagnose time. This is only used to return
                `-np.inf` in case one of the `diag_times` exceeds this value.

            time_dists: Distribution over diagnose times that can be used to
                compute the likelihood of the data, given the spread
                probabilities, but marginalized over the time of diagnosis. If
                set to `None`, a diagnose time must be explicitly set for each
                T-stage.

        Returns:
            The log-likelihood :math:`\\log{p(D \\mid \\theta)}` where :math:`D`
            is the data and :math:`\\theta` is the tuple of spread probabilities
            and diagnose times or distributions over diagnose times.

        See Also:
            :meth:`Unilateral.log_likelihood`: The log-likelihood function of
            the unilateral system.

            :meth:`Bilateral.log_likelihood`: Log-likelihood function of the
            bilateral system, not concerned with midline extension.
        """
        if not self._are_valid_(spread_probs):
            return -np.inf

        self.spread_probs = spread_probs

        llh = 0.

        llh += self.ext._log_likelihood(
            t_stages=t_stages,
            diag_times=diag_times,
            max_t=max_t,
            time_dists=time_dists
        )

        llh += self.noext._log_likelihood(
            t_stages=t_stages,
            diag_times=diag_times,
            max_t=max_t,
            time_dists=time_dists
        )

        return llh


    def marginal_log_likelihood(
        self,
        theta: np.ndarray,
        t_stages: Optional[List[Any]] = None,
        time_dists: dict = {}
    ) -> float:
        """
        Compute the likelihood of the (already stored) data, given the spread
        parameters, marginalized over time of diagnosis via time distributions.
        Wraps the :meth:`log_likelihood` method.

        Args:
            theta: Set of parameters, consisting of the base probabilities
                :math:`b` and the transition probabilities :math:`t`, as well
                as the mixing parameter :math:`\\alpha`. So, it consists of
                these entries:

                +------------+-------------+--------------+
                | base probs | trans probs | mixing param |
                +------------+-------------+--------------+

            t_stages: List of T-stages that should be included in the learning
                process.

            time_dists: Distribution over the probability of diagnosis at
                different times :math:`t` given T-stage.

        Returns:
            The log-likelihood of a parameter sample.

        See Also:
            :meth:`log_likelihood`: Simply calls the actual likelihood function
            where it sets the `diag_times` to `None`.
        """
        return self.log_likelihood(
            theta, t_stages,
            diag_times=None, time_dists=time_dists
        )


    def binom_marg_log_likelihood(
        self,
        theta: np.ndarray,
        t_stages: List[Any],
        max_t: int = 10
    ) -> float:
        """Compute marginal log-likelihood using binomial distributions to sum
        over the diagnose times.

        Args:
            theta: Set of parameters, consisting of the spread probabilities,
                the mixing parameter :math:`\\alpha` and the binomial
                distribution's :math:`p` parameters for each T-category. So,
                its form is

                +------------+-------------+--------------+-----------------+
                | base probs | trans probs | mixing param | binomial params |
                +------------+-------------+--------------+-----------------+

            t_stages: keywords of T-stages that are present in the dictionary of
                C matrices and the previously loaded dataset.

            max_t: Latest accepted time-point.

        Returns:
            The log-likelihood of the (already stored) data, given the spread
            prbabilities as well as the parameters for binomial distribtions
            used to marginalize over diagnose times.
        """
        # splitting theta into spread parameters and...
        len_spread_probs = len(theta) - len(t_stages)
        spread_probs = theta[:len_spread_probs]
        # ...p-values for the binomial distribution
        p = theta[len_spread_probs:]

        if np.any(np.greater(p, 1.)) or np.any(np.less(p, 0.)):
            return -np.inf

        t = np.arange(max_t + 1)
        time_dists = {}
        for i,stage in enumerate(t_stages):
            time_dists[stage] = fast_binomial_pmf(t, max_t, p[i])

        return self.marginal_log_likelihood(
            spread_probs, t_stages,
            time_dists=time_dists
        )


    def risk(self, *args, midline_extension: bool = True, **kwargs) -> float:
        """Compute the risk of nodal involvement given a specific diagnose.

        Args:
            midline_extension: Whether or not the patient's tumor extends over
                the mid-sagittal line.

        See Also:
            :meth:`Bilateral.risk`: Depending on whether or not the patient's
            tumor does extend over the midline, the risk function of the
            respective :class:`Bilateral` instance gets called.
        """
        if midline_extension:
            return self.ext.risk(*args, **kwargs)
        else:
            return self.noext.risk(*args, **kwargs)


    def generate_dataset(
        self,
        num_patients: int,
        stage_dist: List[float],
        ext_prob: float,
        diag_times: Optional[Dict[Any, int]] = None,
        time_dists: Optional[Dict[Any, np.ndarray]] = None,
    ) -> pd.DataFrame:
        """Generate/sample a pandas :class:`DataFrame` from the defined network.

        Args:
            num_patients: Number of patients to generate.
            stage_dist: Probability to find a patient in a certain T-stage.
            ext_prob: Probability that a patient's primary tumor extends over
                the mid-sagittal line.
            diag_times: For each T-stage, one can specify until which time step
                the corresponding patients should be evolved. If this is set to
                ``None``, and a distribution over diagnose times ``time_dists``
                is provided, the diagnose time is drawn from the ``time_dist``.
            time_dists: Distributions over diagnose times that can be used to
                draw a diagnose time for the respective T-stage.
        """
        drawn_ext = np.random.choice(
            [True, False], p=[ext_prob, 1. - ext_prob], size=num_patients
        )
        ext_dataset = self.ext.generate_dataset(
            num_patients=num_patients,
            stage_dist=stage_dist,
            diag_times=diag_times,
            time_dists=time_dists
        )
        noext_dataset = self.noext.generate_dataset(
            num_patients=num_patients,
            stage_dist=stage_dist,
            diag_times=diag_times,
            time_dists=time_dists
        )

        dataset = noext_dataset.copy()
        dataset.loc[drawn_ext] = ext_dataset.loc[drawn_ext]
        dataset[('info', 'tumor', 'midline_extension')] = drawn_ext

        return dataset