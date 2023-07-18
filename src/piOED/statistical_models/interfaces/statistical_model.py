from abc import ABC, abstractmethod

import numpy as np

from ...minimizer.interfaces.minimizer import Minimizer
from ...experiments.interfaces.design_of_experiment import (
    Experiment,
)


class StatisticalModel(ABC):
    """Interface for a statistical model with additional index/experimental design option

    Notation:
    * x: experimental design
    * x0: experiment consisting of experimental design x_0,...,x_N
    * theta: parameter
    * P_theta(x0): probability measure corresponding to the parameter theta and experiment x0


    The specification of an experiment (i.e., a numpy array x0) leads to a statistical model parameterized by theta.
    That is, given theta and x0, we obtain a probability measure denoted P_theta(x0).
    This class contains all the necessary computations required to
    work with a statistical model with respect to the designs of experiment.
    """

    @abstractmethod
    def random(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Draw a random sample of the measurement space corresponding to the measurement P_theta(x0)
        where x0 consists of the single experimental design x

        Parameters
        ----------
        x : np.ndarray
            experimental design
        theta : np.ndarray
            parameter of the statistical model corresponding to x

        Returns
        -------
        np.ndarray
            random drawing from the probability measure P_theta(x)
        """
        pass

    @abstractmethod
    def calculate_fisher_information_matrix(
        self, x0: np.ndarray, theta: np.ndarray
    ) -> np.ndarray:
        # FIXME: need to generalize to experiment not just experimental design.

        """Calculate the Fisher information matrix of the statistical model corresponding to x at the parameter theta

        Parameters
        ----------
        x0 : np.ndarray
            Experiment, i.e. finite number of experimental designs
        theta : np.ndarray
            Parameter of the statistical model corresponding to x

        Returns
        -------
        np.ndarray
            Fisher information matrix of the statistical model corresponding to x at the parameter theta
        """
        pass

    def calculate_cramer_rao_lower_bound(
        self, x0: np.ndarray, theta: np.ndarray
    ) -> np.ndarray:
        """Calculate the Cramer-Rao lower bound of the statistical model corresponding to x at the parameter theta

        Parameters
        ----------
        x0 : np.ndarray
            experiment, i.e. finite number of experimental designs
        theta : np.ndarray
            parameter of the statistical model corresponding to x

        Returns
        -------
        np.ndarray
            Cramer-Rao lower bound of the statistical model corresponding to x at the parameter theta
        """
        return np.linalg.pinv(
            self.calculate_fisher_information_matrix(x0=x0, theta=theta),
            # + 2e-7 * np.identity(len(theta)),
            hermitian=True,
        )

    def optimize_cramer_rao_lower_bound(
            self, x: np.ndarray, previous_experiment: Experiment, theta: np.ndarray, number_designs: int, length: int, index: int
    ) -> float:
        """Calculate the Cramer-Rao lower bound of the statistical model corresponding to x at the parameter theta

        Parameters
        ----------
        x : np.ndarray
            new experiment, i.e. argument of optimization problem
        previous_design: DesignOfExperiment
            experiment of prior iterations
        theta : np.ndarray
            parameter of the statistical model corresponding to x
        number_designs: int
            The number of experimental experiment over which the maximization is taken
        length: int
            The number of independent variables of the design
        index: int
            Index of the diagonal entry of the CRLB to be optimized

        Returns
        -------
        float
            Specified entry (column, row) of Cramer-Rao lower bound of the statistical model corresponding to x at the parameter theta
        """

        if previous_experiment is None:
            x0 = x.reshape(number_designs, length)
        else:
            # If we want to consider an initial experiment within our calculation of the CRLB.
            x0 = np.concatenate(
                    (
                        previous_experiment.experiment,
                        x.reshape(number_designs, length),
                    ),
                    axis=0,
                )
        return np.linalg.pinv(
            self.calculate_fisher_information_matrix(x0=x0, theta=theta),
            #+ 2e-7 * np.identity(len(theta)),
            hermitian=True
        )[index, index]

    def calculate_determinant_fisher_information_matrix(
        self, x0: np.ndarray, theta: np.ndarray
    ) -> float:
        """Calculate the determinant of the Fisher information matrix
        of the statistical model corresponding to x at the parameter theta

        Parameters
        ----------
        x0 : np.ndarray
            experiment, i.e. finite number of experimental designs
        theta : np.ndarray
            parameter of the statistical model corresponding to x

        Returns
        -------
        float
            determinant of the Fisher information matrix of the statistical
            model corresponding to x at the parameter theta
        """
        return np.linalg.det(self.calculate_fisher_information_matrix(x0, theta))

    def optimize_determinant_fisher_information_matrix(
            self, x: np.ndarray, previous_experiment: Experiment, theta: np.ndarray, number_designs: int, length: int
    ) -> float:
        """Calculate the determinant of the Fisher information matrix
        ...of the statistical model corresponding to x at the parameter theta

        Parameters
        ----------
        x : np.ndarray
            new experiment, i.e. argument of optimization problem
        previous_experiment: Experiment
            experiment of prior iterations
        theta : np.ndarray
            parameter of the statistical model corresponding to x
        number_designs: int
            The number of experimental experiment over which the maximization is taken
        length: int
            The number of independent variables of the design

        Returns
        -------
        float
            determinant of the Fisher information matrix of the statistical
            model corresponding to x at the parameter theta
        """
        if previous_experiment is None:
            x0 = x.reshape(number_designs, length)
        else:
            # If we want to consider an initial experiment within our calculation of the CRLB.
            x0 = np.concatenate(
                (
                    previous_experiment.experiment,
                    x.reshape(number_designs, length),
                ),
                axis=0,
            )

        return -np.linalg.det(self.calculate_fisher_information_matrix(x0, theta))

    @abstractmethod
    def calculate_likelihood(
        self, x0: np.ndarray, y: np.ndarray, theta: np.ndarray
    ) -> float:
        """Evaluate the likelihood function of P_theta(x) at y

        Parameters
        ----------
        x0 : np.ndarray
            experiment, i.e. finite number of experimental designs
        y: np.ndarray
            Element of sample space of the probability measure P_theta(x)
        theta : np.ndarray
            parameter of the statistical model corresponding to x

        Returns
        -------
        float
            the likelihood function p_theta(x)(y) of P_theta(x) at y
        """
        pass

    def calculate_maximum_likelihood_estimation(
        self,
        x0: np.ndarray,
        y: np.ndarray,
        minimizer: Minimizer,
    ) -> np.ndarray:
        """Calculate the maximum likelihood estimate of the statistical model corresponding to the experiment x0 at y

        Parameters
        ----------
        x0 : np.ndarray
            experiment, i.e. finite number of experimental designs
        y: np.ndarray
            Element of sample space of the probability measure P_theta(x)
        minimizer : Minimizer
            Minimizer used to maximize the likelihood function at y

        Returns
        -------
        np.ndarray
            the found maximum likelihood parameter estimate for the parameter theta

        """
        return minimizer(
            function=lambda theta: -self.calculate_likelihood(theta=theta, y=y, x0=x0),
            lower_bounds=self.lower_bounds_theta,
            upper_bounds=self.upper_bounds_theta,
        )

    @property
    @abstractmethod
    def lower_bounds_theta(self) -> np.ndarray:
        """Lower bound for parameter theta

        Returns
        -------
        np.ndarray
            Lower bounds for the parameter theta with each entry representing
            the lower bound for the respective entry of theta
        """
        pass

    @property
    @abstractmethod
    def upper_bounds_theta(self) -> np.ndarray:
        """Upper bound for parameter theta

        Returns
        -------
        np.ndarray
            Upper bounds for the parameter theta with each entry representing
            the upper bound of the respective entry of theta
        """
        pass

    @property
    @abstractmethod
    def lower_bounds_x(self) -> np.ndarray:
        """Lower bounds for experimental designs

        Returns
        -------
        np.ndarray
            Lower bounds for an experimental design x with each entry representing
            the lower bound of the respective entry of x
        """
        pass

    @property
    @abstractmethod
    def upper_bounds_x(self) -> np.ndarray:
        """Upper bounds for experimental designs

        Returns
        -------
        np.ndarray
            Upper bounds for an experimental design x with each entry representing
            the upper bound of the respective entry of x
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the statistical model

        Returns
        -------
        str
            Name of the statistical model
        """
        pass
