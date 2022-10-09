import numpy as np

#######
from piOED.parametric_function_library.interfaces.parametric_function import (
    ParametricFunction,
)
from piOED.visualization.plotting_functions import styled_figure, line_scatter


# aging model


class Aging_Model_Cyc:
    def __init__(
            self,
            SoCref: float = 0.5,
            Tref: float = 296.15,
            EOL_C: float = 0.9,
            
            DoDref: float = 0.5,
            crate_ch_ref: float = 0.5,
            crate_dch_ref: float = 1,
            FEC_end: float= 2000,
            
            FEC: np.ndarray = np.array([10, 50, 100, 200, 500, 800, 1500, 2000])
    ):

        # define reference values
        self.SoCref = SoCref  # in p.u.
        self.Tref = Tref  # in K
        self.EOL_C = EOL_C  # in p.u.

        self.DoDref = DoDref  # in p.u.
        self.crate_ch_ref = crate_ch_ref  # in p.u.
        self.crate_dch_ref = crate_dch_ref  # in p.u.
        self.FEC_end = FEC_end
        
        self.FEC = FEC



    # Cycle Aging model

    def x_ref_cyc(self, param: float) -> float:
        return ((1 - self.EOL_C) / (self.FEC_end ** param))

    def d_T_cyc(self, T: float, param: np.ndarray) -> float:
        return np.exp(-param[0] * ((1 / T) - (1 / self.Tref)))

    def d_SoC_cyc(self, SoC: float, param: np.ndarray) -> float:
        return ((SoC / self.SoCref) ** (1 / param[0]))

    def d_crate_ch_cyc(self, crate_ch: float, param: np.ndarray) -> float:
        return ((crate_ch / self.crate_ch_ref) ** (1 / param[0]))

    def d_crate_dch_cyc(self, crate_dch: float, param: np.ndarray) -> float:
        return ((crate_dch / self.crate_dch_ref) ** (1 / param[0]))

    def d_DoD_cyc(self, DoD: float, param: np.ndarray) -> float:
        return ((DoD / self.DoDref) ** (1 / param[0]))


    def Cycle_Aging(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
         """
         theta = [theta0, theta1, theta2, theta3, theta4] .parameters of aging model
         x = [crate_ch (0..1), crate_dch (0..1), DoD (0..1), SoCmean (0..1), T (in K), FEC (0,100,..)] .......independent quantities

         :return: Q_loss_cyc
         """
         Q_loss_cyc = (
         			self.x_ref_cyc(theta[5]) 
         			* self.d_T_cyc(x[0], np.array([theta[0]]))
                    * self.d_SoC_cyc(x[1], np.array([theta[1]]))
                    * self.d_crate_ch_cyc(x[2], np.array([theta[2]]))
                    * self.d_crate_dch_cyc(x[3], np.array([theta[3]]))
                    * self.d_DoD_cyc(x[4], np.array([theta[4]]))
                    * self.FEC ** (theta[5])
			)
         #return round(Q_loss_cyc, 6)
         return Q_loss_cyc



    def partial_derivative_Cycle_Aging(self, theta: np.ndarray, x: np.ndarray, index: int) -> np.ndarray:
        """
        theta = [theta0, theta1, theta2, theta3, theta4] .parameters of aging model
        x = [crate_ch (0..1), crate_dch (0..2), DoD (0..1), SoCmean (0..1), T (in K), FEC (0,100,..)] .......independent quantities
        index = 0..5 ....parameter selection for partial derivative

        :return: partial derivative of Cycle Aging capacity model
        """

        if index == 0:
            return -self.Cycle_Aging(theta, x) * ((1 / x[1]) - (1 / self.Tref))

        elif index == 1:
            return -self.Cycle_Aging(theta, x) * np.log(x[1] / self.SoCref) / theta[1] ** 2

        elif index == 2:
            return -self.Cycle_Aging(theta, x) * np.log(x[2] / self.crate_ch_ref) / theta[2] ** 2

        elif index == 3:
            return -self.Cycle_Aging(theta, x) * np.log(x[3] / self.crate_ch_ref) / theta[3] ** 2

        elif index == 4:
            return self.Cycle_Aging(theta, x) * np.log(x[4] / self.DoDref) / theta[4] ** 2

        elif index == 5:
            return self.Cycle_Aging(theta, x) * (np.log(self.FEC_end) - np.log(self.FEC))

        else:
            pass


#############################


class AgingModelCyc(ParametricFunction):
    def __init__(self):
        self._aging_model = Aging_Model_Cyc()

    def __call__(self, theta: np.ndarray, x: np.ndarray) -> np.ndarray:
        return self._aging_model.Cycle_Aging(theta=theta, x=x)

    def partial_derivative(
            self, theta: np.ndarray, x: np.ndarray, parameter_index: int
    ) -> np.ndarray:
        return self._aging_model.partial_derivative_Cycle_Aging(theta=theta, x=x, index=parameter_index, )

    def second_partial_derivative(
            self,
            theta: np.ndarray,
            x: np.ndarray,
            parameter1_index: int,
            parameter2_index: int,
    ) -> np.ndarray:
        raise NotImplementedError
