import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .timemarg import MarginalizorDict
from .unilateral import Unilateral


class Bilateral:
    """Class that models metastatic progression in a lymphatic system
    bilaterally by creating two :class:`Unilateral` instances that are
    symmetric in their connections. The parameters describing the spread
    probabilities however need not be symmetric.

    See Also:
        :class:`Unilateral`: Two instances of this class are created as
        attributes.
    """
    def __init__(
        self,
        graph: Dict[Tuple[str], List[str]],
        base_symmetric: bool = False,
        trans_symmetric: bool = True,
        **_kwargs
    ):
        """Initialize both sides of the network as a :class:`Unilateral`
        instance:

        Args:
            graph: Dictionary of the same kind as for initialization of
                :class:`Unilateral`. This graph will be passed to the
                constructors of two :class:`Unilateral` attributes of this
                class.
            base_symmetric: If ``True``, the spread probabilities of the two
                sides from the tumor(s) to the LNLs will be set symmetrically.
            trans_symmetric: If ``True``, the spread probabilities among the
                LNLs will be set symmetrically.
        """
        self.ipsi   = Unilateral(graph=graph)   # ipsilateral and...
        self.contra = Unilateral(graph=graph)   # ...contralateral network

        self.base_symmetric  = base_symmetric
        self.trans_symmetric = trans_symmetric

        self.contra.diag_time_dists = self.ipsi.diag_time_dists


    def __str__(self):
        """Print info about the structure and parameters of the bilateral
        lymphatic system.
        """
        num_tumors = len(self.ipsi.tumors)
        num_lnls   = len(self.ipsi.lnls)
        string = (
            f"Bilateral lymphatic system with {num_tumors} tumor(s) "
            f"and 2 * {num_lnls} LNL(s).\n"
        )
        string += "Symmetry: "
        string += "base " if self.base_symmetric else ""
        string += "trans\n" if self.trans_symmetric else "\n"
        string += "Ipsilateral:\t" + " ".join([f"{e}" for e in self.ipsi.edges])
        string += "\n"
        string += "Contralateral:\t" + " ".join([f"{e}" for e in self.contra.edges])

        return string


    @property
    def graph(self) -> Dict[Tuple[str], List[str]]:
        """Return the (unilateral) graph that was used to create this network.
        """
        return self.ipsi.graph


    @property
    def system(self):
        """Return a dictionary with the ipsi- & contralateral side's
        :class:`Unilateral` under the keys ``"ipsi"`` and ``"contra"``
        respectively.

        This is needed since in some weak moment, I thought it would be a great
        idea if a class named ``BilateralSystem`` had an attriute called
        ``system`` which contained two instances of the ``System`` class under
        the keys ``"ipsi"`` and ``"contra"``...
        """
        return {
            "ipsi"  : self.ipsi,
            "contra": self.contra
        }


    @property
    def state(self) -> np.ndarray:
        """
        Return the currently state (healthy or involved) of all LNLs in the
        system.
        """
        ipsi_state = self.ipsi.state
        contra_state = self.contra.state
        return np.concatenate([ipsi_state, contra_state])


    @state.setter
    def state(self, newstate: np.ndarray):
        """
        Set the state of the system to ``newstate``.
        """
        self.ipsi.state = newstate[:len(self.ipsi.lnls)]
        self.contra.state = newstate[len(self.ipsi.lnls):]


    @property
    def base_probs(self) -> np.ndarray:
        """Probabilities of lymphatic spread from the tumor(s) to the lymph
        node levels. If the ipsi- & contralateral spread from the tumor is set
        to be symmetric (``base_symmetric = True``) this only returns the
        parameters of one side. So, the returned array is composed like so:

        +-----------------+--------------------+
        | base probs ipsi | base probs contra* |
        +-----------------+--------------------+

        *Only when ``base_symmetric = False``, which is the default.

        When setting these parameters, the length of the provided array only
        needs to be half as long if ``base_symmetric`` is ``True``, since both
        sides will be set to the same values.

        See Also:
            :attr:`Unilateral.base_probs`
        """
        if self.base_symmetric:
            return self.ipsi.base_probs
        else:
            return np.concatenate([self.ipsi.base_probs,
                                   self.contra.base_probs])

    @base_probs.setter
    def base_probs(self, new_base_probs: np.ndarray):
        """Set the base probabilities from the tumor(s) to the LNLs.
        """
        if self.base_symmetric:
            self.ipsi.base_probs = new_base_probs
            self.contra.base_probs = new_base_probs
        else:
            num_base_probs = len(self.ipsi.base_edges)
            self.ipsi.base_probs = new_base_probs[:num_base_probs]
            self.contra.base_probs = new_base_probs[num_base_probs:]


    @property
    def trans_probs(self) -> np.ndarray:
        """Probabilities of lymphatic spread among the lymph node levels. If
        this ipsi- & contralateral spread is set to be symmetric
        (``trans_symmetric = True``) this only returns the parameters of one
        side. Similiar to the :attr:`base_probs`, this array's shape is:

        +------------------+---------------------+
        | trans probs ipsi | trans probs contra* |
        +------------------+---------------------+

        *Only if ``trans_symmetric = False``.

        And correspondingly, if setting these transmission probability one only
        needs half as large an array if ``trans_symmetric`` is ``True``.

        See Also:
            :attr:`Unilateral.trans_probs`
        """
        if self.trans_symmetric:
            return self.ipsi.trans_probs
        else:
            return np.concatenate([self.ipsi.trans_probs,
                                   self.contra.trans_probs])

    @trans_probs.setter
    def trans_probs(self, new_trans_probs: np.ndarray):
        """Set the transmission probabilities (from LNL to LNL) of the network.
        """
        if self.trans_symmetric:
            self.ipsi.trans_probs = new_trans_probs
            self.contra.trans_probs = new_trans_probs
        else:
            num_trans_probs = len(self.ipsi.trans_edges)
            self.ipsi.trans_probs = new_trans_probs[:num_trans_probs]
            self.contra.trans_probs = new_trans_probs[num_trans_probs:]


    @property
    def spread_probs(self) -> np.ndarray:
        """The parameters representing the probabilities for lymphatic spread
        along a directed edge of the graph representing the lymphatic network.

        If the bilateral network is set to have symmetries, the length of the
        list/array of numbers that need to be provided will be shorter. E.g.,
        when the bilateral lymphatic network is completely asymmetric, it
        requires an array of length :math:`2n_b + 2n_t` where :math:`n_b` is
        the number of edges from the tumor to the LNLs and :math:`n_t` the
        number of edges among the LNLs.

        Similar to the :attr:`base_probs` and the :attr:`trans_probs`, we can
        describe its shape like this:

        +-----------------+--------------------+------------------+----------------------+
        | base probs ipsi | base probs contra* | trans probs ipsi | trans probs contra** |
        +-----------------+--------------------+------------------+----------------------+

        | *Only if ``base_symmetric = False``, which is the default.
        | **Only if ``trans_symmetric = False``.

        See Also:
            :attr:`Unilateral.spread_probs`
        """
        return np.concatenate([self.base_probs, self.trans_probs])

    @spread_probs.setter
    def spread_probs(self, new_spread_probs: np.ndarray):
        """Set the spread probabilities of the :class:`Edge` instances in the
        the network.
        """
        num_base_probs = len(self.ipsi.base_edges)

        if self.base_symmetric:
            self.base_probs = new_spread_probs[:num_base_probs]
            self.trans_probs = new_spread_probs[num_base_probs:]
        else:
            self.base_probs = new_spread_probs[:2*num_base_probs]
            self.trans_probs = new_spread_probs[2*num_base_probs:]


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
        return self.ipsi.diag_time_dists

    @diag_time_dists.setter
    def diag_time_dists(self, new_dists: Union[dict, MarginalizorDict]):
        """Assign new :class:`MarginalizorDict` to this property. If it is a normal
        Python dictionary, try to convert it into a :class:`MarginalizorDict`.
        """
        self.ipsi.diag_time_dists = new_dists
        self.contra.diag_time_dists = self.ipsi.diag_time_dists


    @property
    def modalities(self):
        """Compute the two system's observation matrices
        :math:`\\mathbf{B}^{\\text{i}}` and :math:`\\mathbf{B}^{\\text{c}}`.

        See Also:
            :meth:`Unilateral.modalities`: Setting modalities in unilateral
            System.
        """
        ipsi_modality_spsn = self.ipsi.modalities
        if ipsi_modality_spsn != self.contra.modalities:
            raise RuntimeError(
                "Ipsi- & contralaterally stored modalities are not the same"
            )

        return ipsi_modality_spsn


    @modalities.setter
    def modalities(self, modality_spsn: Dict[str, List[float]]):
        """
        Given specificity :math:`s_P` & sensitivity :math:`s_N` of different
        diagnostic modalities, compute the system's two observation matrices
        :math:`\\mathbf{B}_i` and :math:`\\mathbf{B}_c`.
        """
        self.ipsi.modalities = modality_spsn
        self.contra.modalities = modality_spsn


    @property
    def patient_data(self):
        """Table with rows of patients. Columns should have three levels. The
        first column is ('info', 'tumor', 't_stage'). The rest of the columns
        are separated by modality names on the top level, then subdivided into
        'ipsi' & 'contra' by the second level and finally, in the third level,
        the names of the lymph node level are given. Here is an example of such
        a table:

        +---------+----------------------+----------------------+
        |  info   |         MRI          |         PET          |
        +---------+----------+-----------+----------+-----------+
        |  tumor  |   ipsi   |  contra   |   ipsi   |  contra   |
        +---------+----------+-----------+----------+-----------+
        | t_stage |    II    |    II     |    II    |    II     |
        +=========+==========+===========+==========+===========+
        | early   | ``True`` | ``None``  | ``True`` | ``False`` |
        +---------+----------+-----------+----------+-----------+
        | late    | ``None`` | ``None``  | ``None`` | ``None``  |
        +---------+----------+-----------+----------+-----------+
        | early   | ``True`` | ``False`` | ``True`` | ``True``  |
        +---------+----------+-----------+----------+-----------+
        """
        try:
            return self._patient_data
        except AttributeError as att_err:
            raise AttributeError(
                "No patient data has been loaded yet"
            ) from att_err

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
        mode: str = "HMM"
    ):
        """Load a dataset by converting it into internal representation as data
        matrix.

        Args:
            data: Table with rows of patients. Columns must have three levels.
                The first column is ('info', 'tumor', 't_stage'). The rest of
                the columns are separated by modality names on the top level,
                then subdivided into 'ipsi' & 'contra' by the second level and
                finally, in the third level, the names of the lymph node level
                are given. Here is an example of such a table:

                +---------+---------------------+-----------------------+
                |  info   |         MRI         |         PET           |
                +---------+----------+----------+-----------+-----------+
                |  tumor  |   ipsi   |  contra  |   ipsi    |  contra   |
                +---------+----------+----------+-----------+-----------+
                | t_stage |    II    |    II    |    II     |    II     |
                +=========+==========+==========+===========+===========+
                | early   | ``True`` | ``None`` | ``True``  | ``False`` |
                +---------+----------+----------+-----------+-----------+
                | late    | ``None`` | ``None`` | ``False`` | ``False`` |
                +---------+----------+----------+-----------+-----------+
                | early   | ``True`` | ``True`` | ``True``  | ``None``  |
                +---------+----------+----------+-----------+-----------+

        See Also:
            :meth:`Unilateral.load_data`: Data loading method of unilateral
            system.
        """
        # split the DataFrame into two, one for ipsi-, one for contralateral
        ipsi_data = data.drop(
            columns=["contra"], axis=1, level=1, inplace=False
        )
        ipsi_data = pd.DataFrame(
            ipsi_data.values,
            index=ipsi_data.index,
            columns=ipsi_data.columns.droplevel(1)
        )
        contra_data = data.drop(
            columns=["ipsi"], axis=1, level=1, inplace=False
        )
        contra_data = pd.DataFrame(
            contra_data.values,
            index=contra_data.index,
            columns=contra_data.columns.droplevel(1)
        )

        self.ipsi.load_data(
            ipsi_data,
            modality_spsn=modality_spsn,
            mode=mode
        )
        self.contra.load_data(
            contra_data,
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
            self.diag_time_dists.update(new_marg_params)
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


    def _likelihood(
        self,
        log: bool = True
    ) -> float:
        """Compute the (log-)likelihood of data, using the stored spread probs and
        fixed distributions for marginalizing over diagnose times.

        This method mainly exists so that the checking and assigning of the
        spread probs can be skipped.
        """
        stored_t_stages = set(self.ipsi.diagnose_matrices.keys())
        provided_t_stages = set(self.ipsi.diag_time_dists.keys())
        t_stages = list(stored_t_stages.intersection(provided_t_stages))

        max_t = self.diag_time_dists.max_t
        state_probs = {}
        state_probs["ipsi"] = self.ipsi._evolve(t_last=max_t)
        state_probs["contra"] = self.contra._evolve(t_last=max_t)

        llh = 0. if log else 1.
        for stage in t_stages:
            joint_state_probs = (
                state_probs["ipsi"].T
                @ np.diag(self.ipsi.diag_time_dists[stage].pmf)
                @ state_probs["contra"]
            )
            p = np.sum(
                self.ipsi.diagnose_matrices[stage]
                * (joint_state_probs
                    @ self.contra.diagnose_matrices[stage]),
                axis=0
            )
            if log:
                llh += np.sum(np.log(p))
            else:
                llh *= np.prod(p)

        return llh


    def likelihood(
        self,
        data: Optional[pd.DataFrame] = None,
        given_params: Optional[np.ndarray] = None,
        log: bool = True,
    ):
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

            :meth:`Unilateral.likelihood`: The (log-)likelihood function of
            the unilateral system.
        """
        if data is not None:
            self.patient_data = data

        if given_params is None:
            return self._likelihood(log)

        try:
            self.check_and_assign(given_params)
        except ValueError:
            return -np.inf if log else 0.

        return self._likelihood(log)


    def risk(
        self,
        involvement: Optional[dict] = None,
        given_params: Optional[np.ndarray] = None,
        given_diagnoses: Optional[dict] = None,
        t_stage: str = "early",
        **_kwargs,
    ) -> float:
        """Compute risk of ipsi- & contralateral involvement given specific (but
        potentially incomplete) diagnoses for each side of the neck.

        Args:
            involvement: Nested dictionary that can have keys ``"ipsi"`` and
                ``"contra"``, indicating the respective side's involvement patterns
                that we're interested in. The corresponding values are dictionaries as
                the :class:`Unilateral` model expects them.

            given_params: The risk is a function of these parameters. They mainly
                consist of the :attr:`spread_probs` of the model. Any excess parameters
                will be used to update the parametrized distributions used for
                marginalizing over the diagnose times (see :attr:`diag_time_dists`).

            given_diagnoses: Nested dictionary with keys of diagnostic modalities and
                the values are dictionaries of the same format as the ``involvement``
                arguments.

            t_stage: The T-stage for which the risk should be computed. The attribute
                :attr:`diag_time_dists` must have a distribution for marginalizing
                over diagnose times stored for this T-stage.

        See Also:
            :meth:`Unilateral.risk`: The risk function for only one-sided models.
        """
        if given_params is not None:
            self.check_and_assign(given_params)

        if given_diagnoses is None:
            first_modality = list(self.modalities.keys())[0]
            given_diagnoses = {first_modality: {}}

        for val in given_diagnoses.values():
            if "ipsi" not in val:
                val["ipsi"] = {}
            if "contra" not in val:
                val["contra"] = {}

        diagnose_probs = {}   # vectors containing P(Z=z|X) for respective side
        state_probs = {}      # matrices containing P(X|t) for each side

        for side in ["ipsi", "contra"]:
            side_model = getattr(self, side)
            diagnose_probs[side] = np.zeros(shape=len(side_model.state_list))
            side_diagnose = {mod: diag[side] for mod,diag in given_diagnoses.items()}
            for i,state in enumerate(side_model.state_list):
                side_model.state = state
                diagnose_probs[side][i] = side_model.comp_diagnose_prob(side_diagnose)

            max_t = self.diag_time_dists.max_t
            state_probs[side] = self.system[side]._evolve(t_last=max_t)

        # time-prior in diagnoal matrix form
        time_marg_matrix = np.diag(self.ipsi.diag_time_dists[t_stage].pmf)

        # joint probability P(Xi,Xc) (marginalized over time). Acts as prior
        # for p( Di,Dc | Xi,Xc ) and should be a 2D matrix.
        marg_state_probs = (
            state_probs["ipsi"].T @ time_marg_matrix @ state_probs["contra"]
        )

        # joint probability P(Di=di,Dc=dc, Xi,Xc) of all hidden states and the
        # provided diagnoses
        joint_diag_state = np.einsum(
            "i,ij,j->ij",
            diagnose_probs["ipsi"], marg_state_probs, diagnose_probs["contra"],
        )
        # marginalized probability P(Di=di, Dc=dc)
        marg_diagnose_prob = np.sum(joint_diag_state)

        # P(Xi,Xc | Di=di,Dc=dc)
        post_state_probs = joint_diag_state / marg_diagnose_prob

        if involvement is None:
            return post_state_probs
        if "ipsi" not in involvement:
            involvement["ipsi"] = {}
        if "contra" not in involvement:
            involvement["contra"] = {}

        marg_states = {}   # vectors marginalizing over only the states we care about
        for side in ["ipsi", "contra"]:
            if isinstance(involvement[side], dict):
                involvement[side] = np.array(
                    [involvement[side].get(lnl.name, None) for lnl in side_model.lnls]
                )
            else:
                involvement[side] = np.array(involvement[side])

            side_model = getattr(self, side)
            marg_states[side] = np.zeros(shape=len(side_model.state_list), dtype=bool)
            for i,state in enumerate(side_model.state_list):
                marg_states[side][i] = np.all(np.equal(
                    involvement[side], state,
                    where=(involvement[side] != None),
                    out=np.ones_like(state, dtype=bool)
                ))

        return marg_states["ipsi"] @ post_state_probs @ marg_states["contra"]



    def generate_dataset(
        self,
        num_patients: int,
        stage_dist: Dict[str, float],
    ) -> pd.DataFrame:
        """Generate/sample a pandas :class:`DataFrame` from the defined network.

        Args:
            num_patients: Number of patients to generate.
            stage_dist: Probability to find a patient in a certain T-stage.
        """
        drawn_t_stages, drawn_diag_times = self.diag_time_dists.draw(
            dist=stage_dist, size=num_patients
        )

        drawn_obs_ipsi = self.ipsi._draw_patient_diagnoses(drawn_diag_times)
        drawn_obs_contra = self.contra._draw_patient_diagnoses(drawn_diag_times)
        drawn_obs = np.concatenate([drawn_obs_ipsi, drawn_obs_contra], axis=1)

        # construct MultiIndex for dataset from stored modalities
        sides = ["ipsi", "contra"]
        modalities = list(self.modalities.keys())
        lnl_names = [lnl.name for lnl in self.ipsi.lnls]
        multi_cols = pd.MultiIndex.from_product([sides, modalities, lnl_names])

        # create DataFrame
        dataset = pd.DataFrame(drawn_obs, columns=multi_cols)
        dataset = dataset.reorder_levels(order=[1, 0, 2], axis="columns")
        dataset = dataset.sort_index(axis="columns", level=0)
        dataset[('info', 'tumor', 't_stage')] = drawn_t_stages

        return dataset


class BilateralSystem(Bilateral):
    """Class kept for compatibility after renaming to :class:`Bilateral`.

    See Also:
        :class:`Bilateral`
    """
    def __init__(self, *args, **kwargs):
        msg = ("This class has been renamed to `Bilateral`.")
        warnings.warn(msg, DeprecationWarning)

        super().__init__(*args, **kwargs)