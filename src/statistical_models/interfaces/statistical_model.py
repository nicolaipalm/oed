from abc import ABC, abstractmethod

import numpy as np

from src.minimizer.interfaces.minimizer import Minimizer


class StatisticalModel(ABC):
    """Interface for a statistical model with additional index/experimental designs option

    Notation:
    x: experimental designs
    x0: experiment consisting of experimental designs x
    theta: parameter
    P_theta(x0): probability measure corresponding to the parameter theta and experiment x0


    The specification of an experiment (i.e., a numpy array x0) leads to a statistical model parameterized by theta.
    That is, given theta and x0, we obtain a probability measure denoted P_theta(x0).
    This class contains all the necessary computations required to
    work with a statistical model with respect to the designs of experiments.
    """

    @abstractmethod
    def random(self, x: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Draw a random sample of the measurement space corresponding to the measurement P_theta(x0)
        ...where x0 consists of the single experimental designs x

        Parameters
        ----------
        x : np.ndarray
            experimental designs
        theta : np.ndarray
            parameter of the statistical model corresponding to x

        Returns
        -------
        np.ndarray
            random drawing from the probability measure P_theta(x)
        """
        raise NotImplementedError

    @abstractmethod
    def calculate_fisher_information_matrix(
            self, x0: np.ndarray, theta: np.ndarray
    ) -> np.ndarray:
        # FIXME: need to generalize to experiments not just experimental designs.

        """Calculate the Fisher information matrix of the statistical model corresponding to x at the parameter theta

        Parameters
        ----------
        x0 : np.ndarray
            experiment, i.e. finite number of experimental designs
        theta : np.ndarray
            parameter of the statistical model corresponding to x

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
            hermitian=True
        )

    def calculate_determinant_fisher_information_matrix(
            self, x0: np.ndarray, theta: np.ndarray
    ) -> float:
        """Calculate the determinant of the Fisher information matrix
        ...of the statistical model corresponding to x at the parameter theta

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
            Lower bounds for an experimental designs x with each entry representing
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
            Upper bounds for an experimental designs x with each entry representing
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
