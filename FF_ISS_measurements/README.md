# Summery on the measurements

## Measurement Setup

* Anritsu VectorStar with mm-wave heads up to 150GHz.
* FormFactor SUMMIT200 probe station.
* FormFactor GSG ACP-145 probes with 150um pitch.
* FormFactor GSG impedance standard substrate (ISS) 104-783.

## Data Formate

The measurements are wave parameters stored in `.s2p` touchstone file type. For each measurement sweep there are two files associated to it. The file with the suffix `_A_.s2p` contain the a-waves and the file with suffix `_B_.s2p` contain the b-waves.

As `.s2p` files store 4 complex-valued parameters, the formate for the a-waves and b-waves are given as below:

$$
\mathbf{A} = \begin{bmatrix} a_{11} & a_{12}\\ a_{21} & a_{22}\end{bmatrix}; \qquad \mathbf{B} = \begin{bmatrix} b_{11} & b_{12}\\ b_{21} & b_{22}\end{bmatrix},
$$
where the indices _ij_ for both wave parameters indicate the _i_-th receiver, when excited by the _j_-th port. Remember, there are two ports, and two receivers for each wave parameter.

The S-parameters are calculated as follows:
$$
\mathbf{S} = \mathbf{B}\mathbf{A}^{-1}
$$

## mTRL CPW Standards

The measured standards comprises of coplanar waveguide (CPW) lines and an open standards. The edge-to-edge length of the line standards are given below:

* thru: 200um
* line01: 450um
* line02: 900um
* line03: 1800um
* line04: 3500um
* line05: 5250um

The measured open standard was realized by setting the probes float. This has the effect of -100um offset from the center of the thru standard.