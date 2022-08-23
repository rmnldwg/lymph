from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .bilateral import Bilateral
from .timemarg import MarginalizorDict


class MidlineBilateral:
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
        graph: Dict[Tuple[str], List[str]],
        use_mixing: bool = True,
        trans_symmetric: bool = True,
        **_kwargs
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
            use_mixing: Describe the contralateral base spread probabilities for the
                case of a midline extension as a linear combination between the base
                spread probs of the ipsilateral side and the ones of the contralateral
                side when no midline extension is present.
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
        self.use_mixing = use_mixing
        if self.use_mixing:
            self.alpha_mix = 0.

        self.noext.diag_time_dists = self.ext.diag_time_dists


    @property
    def graph(self) -> Dict[Tuple[str], List[str]]:
        """Return the (unilateral) graph that was used to create this network.
        """
        return self.noext.graph


    @property
    def base_probs(self) -> np.ndarray:
        """Base probabilities of metastatic lymphatic spread from the tumor(s)
        to the lymph node levels. This will return the following concatenation of
        base spread probs depending on whether ``use_mixing`` is set to ``True`` or
        ``False``:

        With the use of a mixing parameter:
        +-----------+-------------------------+--------------+
        | base ipsi | base contra (no midext) | mixing param |
        +-----------+-------------------------+--------------+

        Without it:
        +-----------+----------------------+-------------------------+
        | base ipsi | base contra (midext) | base contra (no midext) |
        +-----------+----------------------+-------------------------+

        When setting these, one needs to provide the respective shape.
        """
        if self.use_mixing:
            return np.concatenate([
                self.ext.ipsi.base_probs,
                self.noext.contra.base_probs,
                [self.alpha_mix],
            ])
        else:
            return np.concatenate([
                self.ext.ipsi.base_probs,
                self.ext.contra.base_probs,
                self.noext.contra.base_probs,
            ])

    @base_probs.setter
    def base_probs(self, new_params: np.ndarray):
        """Set the base probabilities from the tumor(s) to the LNLs, accounting
        for the mixing parameter :math:`\\alpha``.
        """
        k = len(self.ext.ipsi.base_probs)

        self.ext.ipsi.base_probs = new_params[:k]
        self.noext.ipsi.base_probs = new_params[:k]

        if self.use_mixing:
            self.noext.contra.base_probs = new_params[k:2*k]
            self.alpha_mix = new_params[-1]
            # compute linear combination
            self.ext.contra.base_probs = (
                self.alpha_mix * self.ext.ipsi.base_probs
                + (1. - self.alpha_mix) * self.noext.contra.base_probs
            )
        else:
            self.ext.contra.base_probs = new_params[k:2*k]
            self.noext.contra.base_probs = new_params[2*k:3*k]

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

        It is composed of the base spread probs (possible with the mixing parameter)
        and the probabilities of spread among the LNLs.

        +-------------+-------------+
        | base probs  | trans probs |
        +-------------+-------------+
        """
        return np.concatenate([self.base_probs, self.trans_probs])

    @spread_probs.setter
    def spread_probs(self, new_params: np.ndarray):
        """Set the new spread probabilities and the mixing parameter
        :math:`\\alpha`.
        """
        num_base_probs = len(self.base_probs)

        self.base_probs  = new_params[:num_base_probs]
        self.trans_probs = new_params[num_base_probs:]


    @property
    def diag_time_dists(self) -> MarginalizorDict:
        """This property holds the probability mass functions for marginalizing over
        possible diagnose times for each T-stage.

        When setting this property, one may also provide a normal Python dict, in
        which case it tries to convert it to a :class:`MarginalizorDict`.

        Note that the method will provide the same instance of this
        :class:`MarginalizorDict` to both sides of the network.

        See Also:
            :class:`MarginalzorDict`, :class:`Marginalizor`.
        """
        return self.ext.diag_time_dists

    @diag_time_dists.setter
    def diag_time_dists(self, new_dists: Union[dict, MarginalizorDict]):
        """Assign new :class:`MarginalizorDict` to this property. If it is a normal
        Python dictionary, tr to convert it into a :class:`MarginalizorDict`.
        """
        self.ext.diag_time_dists = new_dists
        self.noext.diag_time_dists = self.ext.diag_time_dists


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
            modality_spsn=modality_spsn,
            mode=mode
        )
        self.noext.load_data(
            noext_data,
            modality_spsn=modality_spsn,
            mode=mode
        )


    def check_and_assign(self, new_params: np.ndarray):
        """Check that the spread probability (rates) and the parameters for the
        marginalization over diagnose times are all within limits and assign them to
        the model. Also, make sure the ipsi- and contralateral distributions over
        diagnose times are the same instance of the :class:`MarginalizorDict`.

        Args:
            new_params: The set of :attr:`spread_probs` and parameters to provide for
                updating the parametrized distributions over diagnose times.

        Warning:
            This method assumes that the parametrized distributions (instances of
            :class:`Marginalizor`) all raise a ``ValueError`` when provided with
            invalid parameters.
        """
        k = len(self.spread_probs)
        new_spread_probs = new_params[:k]
        new_marg_params = new_params[k:]

        try:
            self.ext.ipsi.diag_time_dists.update(new_marg_params)
            self.ext.contra.diag_time_dists = self.ext.ipsi.diag_time_dists
            self.noext.ipsi.diag_time_dists = self.ext.ipsi.diag_time_dists
            self.noext.contra.diag_time_dists = self.ext.ipsi.diag_time_dists
        except ValueError as val_err:
            raise ValueError(
                "Parameters for marginalization over diagnose times are invalid"
            ) from val_err

        if new_spread_probs.shape != self.spread_probs.shape:
            raise ValueError(
                "Shape of provided spread parameters does not match network"
            )
        if np.any(0. > new_spread_probs) or np.any(new_spread_probs > 1.):
            raise ValueError(
                "Spread probs must be between 0 and 1"
            )

        self.spread_probs = new_spread_probs


    def likelihood(
        self,
        data: Optional[pd.DataFrame] = None,
        given_params: Optional[np.ndarray] = None,
        log: bool = True,
    ) -> float:
        """Compute log-likelihood of (already stored) data, given the spread
        probabilities and either a discrete diagnose time or a distribution to
        use for marginalization over diagnose times.

        Args:
            data: Table with rows of patients and columns of per-LNL involvment. See
                :meth:`load_data` for more details on how this should look like.

            given_params: The likelihood is a function of these parameters. They mainly
                consist of the :attr:`spread_probs` of the model. Any excess parameters
                will be used to update the parametrized distributions used for
                marginalizing over the diagnose times (see :attr:`diag_time_dists`).

            log: When ``True``, the log-likelihood is returned.

        Returns:
            The log-likelihood :math:`\\log{p(D \\mid \\theta)}` where :math:`D`
            is the data and :math:`\\theta` is the tuple of spread probabilities
            and diagnose times or distributions over diagnose times.

        See Also:
            :attr:`spread_probs`: Property for getting and setting the spread
            probabilities, of which a lymphatic network has as many as it has
            :class:`Edge` instances (in case no symmetries apply).

            :meth:`Unilateral.likelihood`: The log-likelihood function of
            the unilateral system.

            :meth:`Bilateral.likelihood`: The (log-)likelihood function of the
            bilateral system.
        """
        if data is not None:
            self.patient_data = data

        try:
            self.check_and_assign(given_params)
        except ValueError:
            return -np.inf if log else 0.

        llh = 0. if log else 1.

        if log:
            llh += self.ext._likelihood(log=log)
            llh += self.noext._likelihood(log=log)
        else:
            llh *= self.ext._likelihood(log=log)
            llh *= self.noext._likelihood(log=log)

        return llh


    def risk(
        self,
        given_params: Optional[np.ndarray] = None,
        midline_extension: bool = True,
        **kwargs,
    ) -> float:
        """Compute the risk of nodal involvement given a specific diagnose.

        Args:
            spread_probs: Set ot new spread parameters. This also contains the
                mixing parameter alpha in the last position.
            midline_extension: Whether or not the patient's tumor extends over
                the mid-sagittal line.

        See Also:
            :meth:`Bilateral.risk`: Depending on whether or not the patient's
            tumor does extend over the midline, the risk function of the
            respective :class:`Bilateral` instance gets called.
        """
        if given_params is not None:
            self.check_and_assign(given_params)

        if midline_extension:
            return self.ext.risk(**kwargs)
        else:
            return self.noext.risk(**kwargs)


    def generate_dataset(
        self,
        num_patients: int,
        stage_dist: Dict[str, float],
        ext_prob: float,
        **_kwargs,
    ) -> pd.DataFrame:
        """Generate/sample a pandas :class:`DataFrame` from the defined network.

        Args:
            num_patients: Number of patients to generate.
            stage_dist: Probability to find a patient in a certain T-stage.
            ext_prob: Probability that a patient's primary tumor extends over
                the mid-sagittal line.
        """
        drawn_ext = np.random.choice(
            [True, False], p=[ext_prob, 1. - ext_prob], size=num_patients
        )
        ext_dataset = self.ext.generate_dataset(
            num_patients=num_patients,
            stage_dist=stage_dist,
        )
        noext_dataset = self.noext.generate_dataset(
            num_patients=num_patients,
            stage_dist=stage_dist,
        )

        dataset = noext_dataset.copy()
        dataset.loc[drawn_ext] = ext_dataset.loc[drawn_ext]
        dataset[('info', 'tumor', 'midline_extension')] = drawn_ext

        return dataset