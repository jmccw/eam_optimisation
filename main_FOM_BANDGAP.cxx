/*
 * 
 * QWI_FEB-26.cxx
 * PY4115 MAJOR RESEARCH PROJECT
 * Copyright 2024 jmccw <jmcc0@DESKTOP-65IEBIH>
 * 
 */

#include <iostream>
#include <string>
#include <stdio.h>
#include <unistd.h>
#include <math.h>
#include <cmath>     // for std::abs
//using namespace std;
#include <stdlib.h>
#include <stdarg.h>
#include <assert.h>
#include <complex.h>
#include <vector>
#include <fftw3.h>
#include <Eigen/Core>
#include <Eigen/Eigenvalues> 
#include <algorithm>
#include <fstream>

#include <sstream>
#include <map>
#include <omp.h>
//#include <random>

//#include "D:/gnuplot-cpp/gnuplot-cpp/gnuplot_i.hpp"
#include "matrix_operations.cxx"

#include "getXY.cpp"

#include <pthread.h> // modify stack size

//~ #include <functional>
//~ #include <future>
//~ #include <thread>
//~ #include <mutex>
//~ #include <queue>
//~ #include <condition_variable>

#define complex _Complex
const double pi = M_PI;

//Global Variables
bool over_ride_offsets = false;
std::vector<double> CBO_override, VBO_override;
double electron_charge = 1.0; //we use eV, so just let this = 1.
double electric_field = 0.0;
double mass_electron = 1.0; //9.109534e-31; [kg]
double h_c = 1240.0; // eV nm
double hbar_sqaured_2m = 3.81; // [eV Amstrong squared]

const int number_steps = 512;
const int max_matrix = 512;
const int ND = 1024;

std::vector<std::vector<int>> getInputs(std::string input){
	std::vector<std::vector<int>> out(2);
    for(int i = 0; i < 4; i++){
		out[0].push_back( stoi( input.substr(i*7, 3) ) );
		out[1].push_back( stoi( input.substr(i*7+3, 4) ) );
		std::cout << out[0][i] << " " << out[1][i];
	}
	std::cout << std::endl;
    return out;
}
 
std::vector<double> shiftEdgeToCenter(const std::vector<double>& input) {
    std::vector<double> shiftedVector = input;
    int halfSize = input.size() / 2;
    
    // Calculate the number of rotations needed to center the edge values
    int rotations = halfSize % input.size();

    // Perform cyclic rotations
    std::rotate(shiftedVector.begin(), shiftedVector.begin() + rotations, shiftedVector.end());

    return shiftedVector;
}

std::vector<double> pad_func_zeros(std::vector<double> func){
	std::vector<double> func_padded(2*func.size());
	for(int i = 0; i<(int)(0.25*func_padded.size()); i++) func_padded[i] = 0.0;
	for(int i = (int)(0.75*func_padded.size()); i<(int)(func_padded.size()); i++) func_padded[i] = 0.0;
	int j = 0;
	for(int i = (int)(0.25*func_padded.size()); i<(int)(0.75*func_padded.size()); i++){
		func_padded[i] = func[j];
		j++;
	}
	return func_padded;
}

std::vector<double> convolution(std::vector<double> function1, std::vector<double> function2){
	//cout << "in convolution\n";
	if((int)function1.size() != (int)function2.size()) throw std::runtime_error("ERROR @ convolution() :: sizes not equal, "+std::to_string((int)function1.size())+" and "+std::to_string((int)function2.size())+"\n");
	if((int)function1.size() % 8 != 0) std::cout << "Possible problem in convolution" << std::endl;
	
	// Padding
	function1 = pad_func_zeros(function1);
	function2 = pad_func_zeros(function2);
	
	const int N = (int)function1.size(); // Size of input signal -  assumes both vectors are same size
    double* input1 = (double*) fftw_malloc(sizeof(double) * N);
    double* input2 = (double*) fftw_malloc(sizeof(double) * N);
    fftw_complex* output1 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    fftw_complex* output2 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);

	// data in + conversion
    for (int i = 0; i < N; ++i) {
        input1[i] = function1[i]; 
        input2[i] = function2[i];
    }

    // Create FFTW plan for forward transform
    fftw_plan plan1 = fftw_plan_dft_r2c_1d(N, input1, output1, FFTW_ESTIMATE);
    fftw_plan plan2 = fftw_plan_dft_r2c_1d(N, input2, output2, FFTW_ESTIMATE);

    // Execute forward transform
    fftw_execute(plan1);
    fftw_execute(plan2);
    
    fftw_complex* conv_mult = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    std::vector<double> result_vec((int)N/2), result_vec_N((int)N); //Return to original size
    for (int i = 0; i < N; ++i) { //BEWARE OF DIVISION BY N HERE !!! I dont fully understand why this has to happen, but it conserves "area"
        conv_mult[i][0] = (output1[i][0] * output2[i][0] - output1[i][1] * output2[i][1])/N; // Real part
        conv_mult[i][1] = (output1[i][0] * output2[i][1] + output1[i][1] * output2[i][0])/N; // Imaginary part 
    }
    double* result = (double*) fftw_malloc(sizeof(double) * N);
    fftw_plan plan3 = fftw_plan_dft_c2r_1d(N, conv_mult, result, FFTW_ESTIMATE);
    fftw_execute(plan3); 
	for (int i = 0; i < N; i++) {
        result_vec_N[i] = result[i];
    }
    result_vec_N = shiftEdgeToCenter(result_vec_N);
    for (int i = 0; i < N/2; i++) {
        result_vec[i] = result_vec_N[(int)(i+N/4)];
    }
    
    std::cout << "OUT " << (int)result_vec.size() << std::endl;

    // Cleanup resources
    fftw_destroy_plan(plan1);
    fftw_free(input1);
    fftw_free(output1);
    fftw_destroy_plan(plan2);
    fftw_free(input2);
    fftw_free(output2);
    
    fftw_free(result);
    fftw_free(conv_mult);

    return result_vec;
}

class Material {
	private:
		std::string name;
		double valenceBand; // eV
		double conductionBand;
		double bandGap; // eV
		double eEffMass; // kg
		double lhEffMass; // kg
		double hhEffMass; // kg

	public:
		// Constructor
		Material(std::string name_, double valenceBand_, double conductionBand_, double bandGap_, double eEffMass_, double lhEffMass_, double hhEffMass_)
			: name(name_), valenceBand(valenceBand_), conductionBand(conductionBand_), bandGap(bandGap_), eEffMass(eEffMass_), lhEffMass(lhEffMass_), hhEffMass(hhEffMass_) {}
		// Default constructor
		Material() : name(""), valenceBand(0.0), conductionBand(0.0), bandGap(0.0), eEffMass(0.0), lhEffMass(0.0), hhEffMass(0.0) {}

		// Getter methods
		std::string getName() const { return name; }
		double getBG() const { return bandGap; } //get band gap // eV
		double getVB() const { return valenceBand; } //get electron affinity // eV
		double getCB() const { return conductionBand; } //get electron affinity // eV
		double getEffectiveMass(int p) const {
			if (p == 0) { // electron
				return eEffMass;
			} else if (p == 1) { // light hole
				return lhEffMass;
			} else if (p == 2) { // heavy hole
				return hhEffMass;
			} else {
				return 0.0;
			}
		}

		void display() const {
			std::cout << "Material: " << name << std::endl;
			std::cout << "Valence Band: " << valenceBand << " eV" << std::endl;
			std::cout << "Conduction Band: " << conductionBand << " eV" << std::endl;
			std::cout << "Band Gap: " << bandGap << " eV" << std::endl;
			std::cout << "Effective Masses" << std::endl;
			for (int i = 0; i <= 2; i++) {
				std::cout << i << " : " << this->getEffectiveMass(i) << " kg" << std::endl;
			}
		}
};

struct Layer {
	private:
		Material material;
		double thickness; // in Amstrong
		
	public:
		// Constructor 
		Layer(Material material_, double thickness_) : material(material_), thickness(thickness_) {}
		
		double getThickness() const { return thickness; }
		Material getMaterial() const { return material; }
		
		void display() {
			std::cout << "Thickness: " << this->getThickness() << " A" << std::endl;
			std::cout << "Material: " << this->getMaterial().getName() << std::endl;
		}
		
};


double VBO(Material material1, Material material2){ //conduction band offset default
	double calc1 = material1.getCB()-material2.getCB(); //1=w 2=b
	double calc2 = material2.getVB()-material1.getVB();
	double calc3 = material1.getBG()-material2.getBG();
	//cout << "VBO: " << calc2 * calc3 / (calc2+calc1) << endl;
	return -calc2 * calc3 / (calc2+calc1);
}
double CBO(Material material1, Material material2){ //valence band offset default
	double calc1 = material1.getCB()-material2.getCB();
	double calc2 = material2.getVB()-material1.getVB();
	double calc3 = material1.getBG()-material2.getBG();
	//cout << "CBO: " << -calc1 * calc3 / (calc2+calc1) << endl;
	return -calc1 * calc3 / (calc2+calc1);
}



class Heterostructure {
	private:
		//vector<Layer> layers;
		double heterostructure_thickness;
		std::vector<std::vector<double> > potential_;
		std::map<std::string, Material> materials; // Map to store materials
		std::vector<Layer> layers;
		
	public:
		// FYP IMPLEMENTAITON | Constructor, reads in material parameters and layer file
		Heterostructure(const std::string& materialFileName, const std::string& layerFileName) : potential_(3, std::vector<double>(number_steps)) {
			// Read material information from the input file
			std::ifstream materialFile(materialFileName);
			std::string line;

			// Read materials line by line
			while (std::getline(materialFile, line)) {
				std::istringstream iss(line);
				std::string materialName;
				double bandGap, valenceBand, conductionBand, eEffMass, lhEffMass, hhEffMass;
				if (!(iss >> materialName >> valenceBand >> conductionBand >> bandGap >> eEffMass >> lhEffMass >> hhEffMass)) {
					throw std::runtime_error("Error: Invalid material file format");
				}
				// Create Material object and store in the map
				Material material(materialName, valenceBand, conductionBand, bandGap, eEffMass, lhEffMass, hhEffMass);
				materials[materialName] = material;
			}

			// Read layer information from the input file
			std::ifstream layerFile(layerFileName);
			int numLayers;
			if (!(layerFile >> numLayers)) {
				throw std::runtime_error("Error: Invalid layer file format");
			}

			// Read layers line by line
			for (int i = 0; i < numLayers; ++i) {
				std::string materialName;
				double thickness;
				if (!(layerFile >> materialName >> thickness)) {
					throw std::runtime_error("Error: Invalid layer file format");
				}
				// Retrieve Material object from the map using the material name
				auto it = materials.find(materialName);
				if (it == materials.end()) {
					throw std::runtime_error("Error: Unknown material name in layer file");
				}
				Material material = it->second;
				// Create Layer object and add to layers vector
				layers.push_back(Layer(material, thickness));
			}
			
			double total_thickness = 0.0;
			for(const Layer& layer : layers) {
				total_thickness += layer.getThickness();
			}
			
			this->heterostructure_thickness = total_thickness;
			resetPotential();
		}
		
		// PhD IMPLEMENTATION | constructor that accepts a vector of Layer objects directly
		Heterostructure(const std::vector<Layer>& inputLayers) : potential_(3, std::vector<double>(number_steps)), layers(inputLayers) {
			// Validate and compute total thickness
			double total_thickness = 0.0;
			for(const Layer& layer : layers) {
				total_thickness += layer.getThickness();
			}
			this->heterostructure_thickness = total_thickness;
			resetPotential();
		}
		
		
		double getPotential(int particle, double x, int x_int) {
			if (particle == 1 || particle == 2){ // if particle is hole
				return this->potential_[particle][x_int] - electron_charge*electric_field*(x-this->getThickness()/2.0); 
			}
			else if (particle == 0){             // else it is electron
				return this->potential_[particle][x_int] + electron_charge*electric_field*(x-this->getThickness()/2.0);
			}
			return potential_[particle][x_int];
		}
		
		std::vector<Layer> getLayers() const {
			return this->layers;
		}
		
		// Getters / Setters
		double getThickness() const {
			return this->heterostructure_thickness;
		}
		
		void resetPotential(){
			double x_set[number_steps];
			double delta_x = this->getThickness()/(number_steps+1);
			for(int p = 0; p < 3; p++){
				for(int i = 0; i < number_steps; i++){
					x_set[i] = delta_x + i*delta_x;
					this->potential_[p][i] = potential(p, x_set[i]);
				}
			}
		}
		
		double eff_mass(int particle, double x) const { //
			double material_threshold = 0.0;
			for(const Layer& layer : layers) {
				material_threshold += layer.getThickness();		
				//Material threshold determines whether or not we need to add a band offset to the potential at x
				if(x <= material_threshold) {			
					return layer.getMaterial().getEffectiveMass(particle);
				}
			}
			throw std::runtime_error("Error: Unable to find effective mass for the given position."+std::to_string(x));
			return 0.0; //this will never happen under normal circumstances
		}
		
		double potential(int particle, double x) { //default
			// particle = 0,1,2 -> elecron, lh, hh
			double material_threshold = 0.0;
			double U = 0.0, V = 0.0;
			int i = 0;

			for(Layer layer : this->layers) {
				material_threshold += layer.getThickness();	
				//Material threshold determines whether or not we need to add a band offset to the potential at x
				if(x >= material_threshold) {
					if(over_ride_offsets == false) {
						// so if x is not within material of layer[i] add/subtract relevant band offset
						U += CBO(this->layers[i].getMaterial(), this->layers[i+1].getMaterial());
						V += VBO(this->layers[i].getMaterial(), this->layers[i+1].getMaterial());						
					} else if (over_ride_offsets == true) {
						U += CBO_override[i]; 
						V += VBO_override[i]; 
					}
					i++;
				}
				else { // x within new material threshold, return relevant potential
					if (particle == 1 || particle == 2){ // if particle is hole
						return V;  
					}
					else if (particle == 0){      // else it is electron
						return U;
					}
				}
			}

			throw std::runtime_error("Error: Unable to find valid potential for the given position."+std::to_string(x));
			return 0.0; //this will never happen under normal circumstances but leaving this here to make compiler happy.
		}
		
		void intermixPotential(double strength) {
			std::vector<double> Gauss_y(number_steps), x_het(number_steps);
			std::vector<std::vector<double> > Potential_y(3, std::vector<double>(number_steps));
			double sum = 0, x_0 = this->getThickness()/2.0, sum_pot_init = 0;
			double sigma = strength;
			for(int p = 0; p < 3; p++){
				for(int i = 0; i < number_steps; i++){
					double delta_x = (this->getThickness()/(number_steps+1));
					x_het[i] = delta_x + i*delta_x;
					Potential_y[p][i] = this->potential(p, x_het[i]);
					Gauss_y[i] = ( 1.0 / (sigma*sqrt(2*pi)) ) * exp( -(x_het[i]-x_0) * (x_het[i]-x_0) / (2.0*sigma*sigma) );
				}
			}
			
			// Normalise Gaussian - so that potential is "CTS"
			for(int i = 0; i < number_steps; i++){
				sum += Gauss_y[i];
				sum_pot_init += Potential_y[0][i];
			}

			for(int i = 0; i < number_steps; i++){
				Gauss_y[i] = Gauss_y[i]/std::abs(sum);///sum;
			}

			//cout << "Conv.\n"; // Defining 3 rows in the case that strain is ever implemented.
			std::vector<std::vector<double> > Potential_QWI(3, std::vector<double>(number_steps));
			Potential_QWI[0] = convolution(Potential_y[0], Gauss_y);
			Potential_QWI[1] = convolution(Potential_y[1], Gauss_y);
			Potential_QWI[2] = convolution(Potential_y[2], Gauss_y);
			
			//cout << "Conv returned.\n";
			//~ Potential_QWI[0] = shiftEdgeToCenter(Potential_QWI[0]);
			//~ Potential_QWI[1] = shiftEdgeToCenter(Potential_QWI[1]);
			//~ Potential_QWI[2] = shiftEdgeToCenter(Potential_QWI[2]);
			
			// printing convolution result
			double sum_pot_QWI = 0;
			for(int i = 0; i<number_steps; i++){
				sum_pot_QWI += Potential_QWI[0][i];
			}
			std::cout << "'Area' before Intermixing > " << sum_pot_init << std::endl;
			std::cout << "'Area' after Intermixing > " << sum_pot_QWI << std::endl;
			for(int p = 0; p < 3; p++){
				this->potential_[p] = Potential_QWI[p];
			}		
		}
		
		void display() { // printing heterostructure details
			std::cout << "Layers: " << std::endl;
			int i = 1;
			for(Layer layer : layers){
				std::cout << i << " : " << layer.getMaterial().getName() << " : " << layer.getThickness() << " A"<< std::endl;
				i++;
			}	
			std::cout << "Total Thickness : " << this->getThickness() << " A" << std::endl;
			std::cout << "\nMaterial properties by layer:"<<std::endl;	
			for(Layer layer : layers) {
				layer.getMaterial().display();
				std::cout << std::endl;
			}	
		}
};

// Material constructors : 

// Decleration: Material(EF, BG, e_eff_mass, lh_eff_mass, hh_eff_mass, refractive index) 
Material GaAs("GaAs", 0.111,1.53, 1.42, 0.063, 0.082, 0.51);
Material GaP("GaP", -0.388,2.5255, 2.74, 0.25, 0.14, 0.67);
Material InP("InP", 0.0,1.35, 1.35, 0.077, 0.14, 0.6);
Material InAs("InAs", 0.441,0.801, 0.354, 0.023, 0.026, 0.41);
Material AlAs("AlAs", -0.4245,2.5255, 2.95, 0.15, 0.16, 0.79);

// materials = [GaAs, GaP, InP, InAs, AlAs];
// Simulation setup :: InGaAlAs
double BG_InGaAlAs(double x, double y){
    return 0.36 + 2.093*y + 0.629*x + 0.577*y*y + 0.436*x*x + 1.013*x*y - 2.0*x*y*(1-x-y); //# [eV]
}
double EF_InGaAlAs(double x, double y){ //# Effective electron finity for placing conduction bands InGaAlAs
    return 0.5766 - 0.3439*BG_InGaAlAs(x, y); //# [eV] 
}
double effMass_InGaAlAs(double x, double y, int particle){
    return InAs.getEffectiveMass(particle)*(1-x-y) + GaAs.getEffectiveMass(particle)*(x) + AlAs.getEffectiveMass(particle)*(y);
}
double VB_InGaAlAs(double x, double y){ //# Effective electron finity for placing conduction bands InGaAlAs
	return (1-x-y)*InAs.getVB() + x*GaAs.getVB() + y*AlAs.getVB(); //# [eV]
}
double CB_InGaAlAs(double x, double y){ //# Effective electron finity for placing conduction bands InGaAlAs
	return (1-x-y)*InAs.getCB() + x*GaAs.getCB() + y*AlAs.getCB(); //# [eV]
}

double relative_energy(double energy, int p, Heterostructure& QW) { // accept a 2 paramters [energy relative to well of..][particle type]
    // corrects an (inital) calculated energy from QW solution for respective band of type particle type p
    double E_REL;
    
    // this should be okay with and without override on
    double BG = abs(QW.getLayers()[0].getMaterial().getBG());
	for (int i = 1; i < (int)QW.getLayers().size() - 1; ++i) {
		double band_gap = abs(QW.getLayers()[i].getMaterial().getBG());
		BG = std::min(BG, band_gap);
	}
	
    if(p==0){
		double max_CBO = CBO(QW.getLayers()[0].getMaterial(), QW.getLayers()[1].getMaterial());
		double pos = CBO(QW.getLayers()[0].getMaterial(), QW.getLayers()[1].getMaterial());
		for (int i = 1; i < (int)QW.getLayers().size()-1; ++i) {
			pos += CBO(QW.getLayers()[i].getMaterial(), QW.getLayers()[i+1].getMaterial());
			max_CBO = std::max(abs(max_CBO), abs(pos));
		}
		//cout << "max_CBO" << max_CBO << endl;
		E_REL = energy + BG + max_CBO;
    } else { // i.e. p==1 or p==2
		double max_VBO = VBO(QW.getLayers()[0].getMaterial(), QW.getLayers()[1].getMaterial());
		double pos = VBO(QW.getLayers()[0].getMaterial(), QW.getLayers()[1].getMaterial());
		for (int i = 1; i < (int)QW.getLayers().size()-1; ++i) {
			pos += VBO(QW.getLayers()[i].getMaterial(), QW.getLayers()[i+1].getMaterial());
			max_VBO = std::max(abs(max_VBO), abs(pos));
		}
		//cout << "max_VBO" << max_VBO << endl;
		E_REL = -(max_VBO) - energy;	
	}
    return E_REL;
}


// Define a function to solve the heterostructure
void solve(Heterostructure& heterostructure, std::vector<double>& x_out,
           std::vector<std::vector<double>>& energies,
           std::vector<std::vector<std::vector<double>>>& eigenVectors) {
			   
	double delta_x = heterostructure.getThickness() / (number_steps + 1.0); 
	double x[number_steps];
	x[0] = delta_x; 
	x_out[0] = x[0]; 

	for(int i = 1; i<number_steps; i++){ //initialise x (once)
		x[i] = x[0] + i*delta_x;
		x_out[i] = x[i];
	}
	
	//implementation of Kowano & Kito : 'Optical Waveguide Analysis' : solution of SE with effective mass approximation for all bands/particles
	#pragma omp parallel for
    for(int p = 0; p<=2; p++){ //for all particle types
		//initialise solution matrix M
		double M[number_steps][number_steps];
		for(int i = 0; i<number_steps; i++){
			for(int j = 0; j<number_steps; j++){
				M[i][j] = 0;
			}
		}
	
		double alpha_w[number_steps], alpha_e[number_steps], alpha_x[number_steps];
		alpha_w[0] = -hbar_sqaured_2m * 1.0/(delta_x*delta_x) * 2.0/(heterostructure.eff_mass(p,x[0])+heterostructure.eff_mass(p,x[0]));
        alpha_e[0] = -hbar_sqaured_2m * 1.0/(delta_x*delta_x) * 2.0/(heterostructure.eff_mass(p,x[0])+heterostructure.eff_mass(p,x[1]));
        alpha_x[0] = -alpha_w[0]-alpha_e[0];
        
        M[0][0] = alpha_x[0] + heterostructure.getPotential(p,x[0],0);
        M[0][1] = alpha_e[0];

        for(int nr = 1; nr < number_steps-1; nr++){
            alpha_w[nr] = -hbar_sqaured_2m * 1.0/(delta_x*delta_x) * 2.0/(heterostructure.eff_mass(p,x[nr])+heterostructure.eff_mass(p,x[nr-1]));
            alpha_e[nr] = -hbar_sqaured_2m * 1.0/(delta_x*delta_x) * 2.0/(heterostructure.eff_mass(p,x[nr])+heterostructure.eff_mass(p,x[nr+1]));
            alpha_x[nr] = -alpha_w[nr]-alpha_e[nr];
            
            		//~ cout << "pot="<<heterostructure.getPotential(p,x[nr],nr) << endl; 


            M[nr][nr-1] = alpha_w[nr];    //sub-diagonal
            M[nr][nr] = alpha_x[nr] + heterostructure.getPotential(p,x[nr],nr); //diagonal
            M[nr][nr+1] = alpha_e[nr];   //upper diagonal   
		}

        alpha_w[number_steps-1] = -hbar_sqaured_2m * 1.0/(delta_x*delta_x) * 2.0/(heterostructure.eff_mass(p,x[number_steps-1])+heterostructure.eff_mass(p,x[number_steps-1-1]));
        alpha_e[number_steps-1] = -hbar_sqaured_2m * 1.0/(delta_x*delta_x) * 2.0/(heterostructure.eff_mass(p,x[number_steps-1])+heterostructure.eff_mass(p,x[number_steps-1])); // assuming m(x_edge-dx) = m(x_edge) as boundary condition
        alpha_x[number_steps-1] = -alpha_w[number_steps-1]-alpha_e[number_steps-1];
        M[number_steps-1][number_steps-2] = alpha_w[number_steps-1];
        M[number_steps-1][number_steps-1] = alpha_x[number_steps-1] + heterostructure.getPotential(p,x[number_steps-1],number_steps-1);

		Eigen::MatrixXf M_eigen(number_steps, number_steps);
		for (int i = 0; i < number_steps; ++i) { // really not a speed issue here
			for (int j = 0; j < number_steps; ++j) {
				M_eigen(i, j) = M[i][j];
			}
		}

		// Solve the matrix using Eigen's EigenSolver
		Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> solver(M_eigen);
		Eigen::VectorXf eigenvalues = solver.eigenvalues();
		Eigen::MatrixXf eigenvectors = solver.eigenvectors();
		for (int i = 0; i < number_steps; ++i) {
			energies[p][i] = eigenvalues(i);			// Assign eigenvalue to energies
			for (int j = 0; j < number_steps; ++j) {	// Assign eigenvector to eigenVectors
				eigenVectors[p][i][j] = eigenvectors(j, i);
			}
		}
	}
}

//~ vector<vector<vector<double>>> findTransitions(vector<vector<double>> energies){
	//~ vector<vector<vector<double>>> E_gap(number_steps, vector<vector<double>>(number_steps, vector<double>(2))); //Gap between [electron state] and [hole state] of [hole type]
	//~ for (int k=0; k<2; k++){ // for 2 hole types
        //~ for (int i=0; i<(int)energies[0].size(); i++){ // for all electrons
            //~ for (int j=0; j<(int)energies[0].size(); j++){ // and all hole states
                //~ E_gap[i][j][k] = abs(energies[0][i]-energies[k+1][j]);
			//~ }
		//~ }
	//~ }
    //~ return E_gap; // a matrix/array of energy transtions indexed as [electron state][hole state][hole type] 
//~ }


//~ vector<vector<double>> findEnergiesRelative(vector<vector<double>> energies, Heterostructure& heterostructure) {
	//~ vector<vector<double>> energies_relative(3, vector<double>(number_steps));
	//~ for (int k=0; k<3; k++){ // for 3 particle types
        //~ for (int i=0; i<(int)energies[0].size(); i++){ //for all states
            //~ energies_relative[k][i] = relative_energy(energies[k][i], k, heterostructure);
		//~ }
	//~ }
	//~ return energies_relative; //returns [3][number_solutions] matrix of energies relative to their ectual value in well structure
//~ }

std::vector<double> overlapIntegral(std::vector<double> vector1, std::vector<double> vector2){
	if((int)vector1.size() != (int)vector2.size()) throw std::runtime_error("vector sizes not equal, "+std::to_string((int)vector1.size())+" and "+std::to_string((int)vector2.size()));
	std::vector<double> overlap((int)vector1.size()), vector1_dummy((int)vector1.size()), vector2_dummy((int)vector2.size());
	double N1 = 0.0, N2 = 0.0;
	// possibly might need to declare some dummy vectors ?? - this was a python issue
	for(int i = 0; i < (int)overlap.size(); i++){
		N1 += fabs(vector1[i])*fabs(vector1[i]);
		N2 += fabs(vector2[i])*fabs(vector2[i]);
	}
	for(int i = 0; i < (int)overlap.size(); i++){
		vector1_dummy[i] = vector1[i]/N1;
		vector2_dummy[i] = vector2[i]/N2;
		overlap[i] = vector1_dummy[i]*vector2_dummy[i];
	}
	return overlap;
}

//~ double I_squared(vector<double> vector1, vector<double> vector2){
	//~ vector<double> overlap = overlapIntegral(vector1, vector2);
	//~ double I_squared = 0;
	//~ for(int i = 0; i < (int)overlap.size(); i++) I_squared += fabs(overlap[i]);
	//~ I_squared *= I_squared; // square result
	//~ return I_squared;
//~ }

//~ vector<vector<vector<double>>> findOverlapsAll(vector<vector<vector<double>>> wavefunctions) {
	//~ vector<vector<vector<double>>> I_squared_matrix(number_steps, vector<vector<double>>(number_steps, vector<double>(2)));
	//~ // [electron state][hole state][hole type]
	//~ for (int k=0; k<2; k++) { // for 2 hole types
        //~ for (int i=0; i<(int)wavefunctions[0].size(); i++) { // for all electrons
			//~ vector<double> state1 = wavefunctions[0][i];
            //~ for (int j=0; j<(int)wavefunctions[0].size(); j++) { // and all hole states
				//~ vector<double> state2 = wavefunctions[1+k][j];
                //~ I_squared_matrix[i][j][k] = I_squared(state1, state2);
			//~ }
		//~ }
	//~ }
	//~ return I_squared_matrix;
//~ }



//~ vector<vector<vector<double>>> wavelengthTransformation(vector<vector<vector<double>>> data_in) {
	//~ //The intention here is that the user passes in E_GAP to find an adjacent matrix in wavelength terms.
	//~ vector<vector<vector<double>>> data_out(number_steps, vector<vector<double>>(number_steps, vector<double>(2)));
	//~ for (std::size_t i = 0; i < data_out.size(); ++i) {
		//~ for (std::size_t j = 0; j < data_out[i].size(); ++j) {
			//~ for (std::size_t k = 0; k < data_out[i][j].size(); ++k) {
				//~ data_out[i][j][k] = h_c / data_in[i][j][k]; //h_c = 1240 eV nm :: so [eV nm / eV] = [nm]
			//~ }
		//~ }
	//~ }
	//~ return data_out;
//~ }

void plot_potential(Heterostructure& QW){
	std::vector<double> potential1(number_steps), potential2(number_steps);
	std::vector<double> x_test(number_steps);
	for(int i = 0; i<number_steps; i++){
		x_test[i] = (QW.getThickness()/(number_steps+1)) + i*(QW.getThickness()/(number_steps+1));
		potential1[i] = relative_energy(QW.getPotential(2, x_test[i],i), 2, QW);
		potential2[i] = relative_energy(QW.getPotential(0, x_test[i],i), 0, QW);
	}
			
	//~ try {
		//~ Gnuplot gp;
		//~ gp.set_title("Wavefunctions");
		//~ gp.set_xlabel("z (angstrom)");
		//~ gp.set_ylabel("mag. (AU)");
		//~ gp.plot_xy(x_test, potential1, "CB");
		//~ gp.plot_xy(x_test, potential2, "VAL");

		//~ // keep plot window open until the user presses Enter
		//~ cout << "Press Enter to close the plot..." << endl;
		//~ cin.get();  // Wait for user input
		//~ cin.get();
	//~ }
	//~ catch (const GnuplotException& ge) {
		//~ cerr << ge.what() << endl;
		//~ //return -1;
	//~ }		
			
}

void plot(std::vector<double> vecx, std::vector<double> vecy) {
	//~ try {
		//~ Gnuplot gp;
		//~ gp.set_title("plot");
		//~ gp.set_xlabel("x");
		//~ gp.set_ylabel("y");
		//~ gp.plot_xy(vecx, vecy, "exe");

		//~ // keep plot window open until the user presses Enter
		//~ cout << "Press Enter to close the plot..." << endl;
		//~ cin.get();  // Wait for user input
		//~ cin.get();
	//~ }
	//~ catch (const GnuplotException& ge) {
		//~ cerr << ge.what() << endl;
		//~ //return -1;
	//~ }		
}

double findGroundStateQWI(Heterostructure& heterostructure, int p) {  
	double delta_x = heterostructure.getThickness() / (number_steps + 1.0); 
	double x[number_steps];
	x[0] = delta_x; 
	//x_out[0] = x[0]; 

	for(int i = 1; i<number_steps; i++){ //initialise x (once)
		x[i] = x[0] + i*delta_x;
		//x_out[i] = x[i];
	}

	//implementation of Kowano & Kito : 'Optical Waveguide Analysis' : solution of SE with effective mass approximation for all bands/particles

	double M[number_steps][number_steps];
	for(int i = 0; i<number_steps; i++){
		for(int j = 0; j<number_steps; j++){
			M[i][j] = 0;
		}
	}

	double alpha_w[number_steps], alpha_e[number_steps], alpha_x[number_steps];
	alpha_w[0] = -hbar_sqaured_2m * 1.0/(delta_x*delta_x) * 2.0/(heterostructure.eff_mass(p,x[0])+heterostructure.eff_mass(p,x[0]));
	alpha_e[0] = -hbar_sqaured_2m * 1.0/(delta_x*delta_x) * 2.0/(heterostructure.eff_mass(p,x[0])+heterostructure.eff_mass(p,x[1]));
	alpha_x[0] = -alpha_w[0]-alpha_e[0];
	
	M[0][0] = alpha_x[0] + heterostructure.getPotential(p,x[0],0);
	M[0][1] = alpha_e[0];

	for(int nr = 1; nr < number_steps-1; nr++){
		alpha_w[nr] = -hbar_sqaured_2m * 1.0/(delta_x*delta_x) * 2.0/(heterostructure.eff_mass(p,x[nr])+heterostructure.eff_mass(p,x[nr-1]));
		alpha_e[nr] = -hbar_sqaured_2m * 1.0/(delta_x*delta_x) * 2.0/(heterostructure.eff_mass(p,x[nr])+heterostructure.eff_mass(p,x[nr+1]));
		alpha_x[nr] = -alpha_w[nr]-alpha_e[nr];

		M[nr][nr-1] = alpha_w[nr];    //sub-diagonal
		M[nr][nr] = alpha_x[nr] + heterostructure.getPotential(p,x[nr],nr); //diagonal
		M[nr][nr+1] = alpha_e[nr];   //upper diagonal  
	}

	alpha_w[number_steps-1] = -hbar_sqaured_2m * 1.0/(delta_x*delta_x) * 2.0/(heterostructure.eff_mass(p,x[number_steps-1])+heterostructure.eff_mass(p,x[number_steps-1-1]));
	alpha_e[number_steps-1] = -hbar_sqaured_2m * 1.0/(delta_x*delta_x) * 2.0/(heterostructure.eff_mass(p,x[number_steps-1])+heterostructure.eff_mass(p,x[number_steps-1])); // assuming m(x_edge-dx) = m(x_edge) as boundary condition
	alpha_x[number_steps-1] = -alpha_w[number_steps-1]-alpha_e[number_steps-1];
	M[number_steps-1][number_steps-2] = alpha_w[number_steps-1];
	M[number_steps-1][number_steps-1] = alpha_x[number_steps-1] + heterostructure.getPotential(p,x[number_steps-1],number_steps-1);
	
	//solve Matrix (using Eigen)
	Eigen::MatrixXd M_eigen(number_steps, number_steps);
	for (int i = 0; i < number_steps; ++i) { // really not a speed issue here
		for (int j = 0; j < number_steps; ++j) {
			M_eigen(i, j) = M[i][j];
		}
	}
	
	// Solve the matrix using Eigen's EigenSolver
	Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(M_eigen);
	Eigen::VectorXd eigenvalues = solver.eigenvalues();
	//Eigen::MatrixXd eigenvectors = solver.eigenvectors();
	std::vector<double> energies(number_steps);
	for (int i = 0; i < number_steps; ++i) { // really not a speed issue here
		energies[i] = eigenvalues(i);
	}
	double w = *std::min_element(energies.begin(), energies.end());
	return relative_energy(w, p, heterostructure);
}

struct SimulationParameters {
    bool intermixingEnabled;
    double targetBandgapShift;
    int numElectricFields;
    double maxElectricField;
};

SimulationParameters readSimulationParameters(const std::string& filename) {
    SimulationParameters params;

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Failed to open file " << filename << std::endl;
        // You might want to handle this error condition appropriately
        // For now, just return default-initialized parameters
        return params;
    }

    std::string intermixing;
    double targetBandgapShift;
    if (!(file >> intermixing >> targetBandgapShift)) {
        std::cerr << "Error: Failed to read intermixing parameters" << std::endl;
        return params;
    }
    params.intermixingEnabled = (intermixing == "TRUE");
    params.targetBandgapShift = targetBandgapShift;

    if (!(file >> params.numElectricFields)) {
        std::cerr << "Error: Failed to read number of electric fields" << std::endl;
        return params;
    }

    if (!(file >> params.maxElectricField)) {
        std::cerr << "Error: Failed to read max electric field" << std::endl;
        return params;
    }

    return params;
}

double simulate(std::vector<Layer> input){
	electric_field = 0.0;
	Heterostructure QW_(input);
	//QW_.display();
	
	std::vector<std::vector<double> > energies(3, std::vector<double>(number_steps)); //particle, energy_level
	std::vector<std::vector<std::vector<double> > > eigenVectors(3, std::vector<std::vector<double> >(number_steps, std::vector<double>(number_steps)));
	std::vector<double> x(number_steps); //length element

	solve(QW_, x, energies, eigenVectors); //PARALELLISED - problem here
	
	double bandgap = abs(relative_energy(energies[0][0], 0, QW_)-relative_energy(energies[2][0],2, QW_));
	//cout << "bandgap=" << abs(relative_energy(energies[0][0], 0, QW_)-relative_energy(energies[2][0],2, QW_));
	return bandgap;
}


//~ int main(int argc, char **argv) /* Just prints bandgap */
//~ {
	//~ pthread_attr_t attr;
    //~ pthread_attr_init(&attr);
    //~ pthread_attr_setstacksize(&attr, 16 * 1024 * 1024); // 16 MB allocated to stack
    
	//~ omp_set_num_threads(4);
	//~ Eigen::setNbThreads(4);
	//~ Eigen::initParallel();
		
	//~ std::vector<Layer> input_layers;
	
	//~ /* provide final chromosome information here to test Bandgap - example here is just random */
	//~ std::vector<std::vector<double> > chromosome = {{0.414,0.243},{0.728,0.184},{0.551,0.655}};
	
	//~ for(int i = 0; i < 3; i++){
		//~ double x, y;
		//~ GetXY( 0.8+(0.68675*(chromosome[i][1])), 0.0, x, y);
		//~ std::cout << (chromosome[i][0]) << " " << chromosome[i][1] <<std::endl;
		//~ Material material(std::to_string(i), VB_InGaAlAs(x,y), CB_InGaAlAs(x,y), BG_InGaAlAs(x,y), effMass_InGaAlAs(x,y,0), effMass_InGaAlAs(x,y,1), effMass_InGaAlAs(x,y,2));
		//~ input_layers.push_back(Layer(material,30.0+90.0*(double)chromosome[i][0]));
	//~ }
	
	//~ double bandgap = simulate(input_layers);
	//~ std::cout << bandgap << "eV = " << h_c/bandgap << " nm." << std::endl;
	
	//~ return 0;
//~ }

