/**********************************************************************
  attempt1.cxx
 **********************************************************************

  Test program for use of GAUL adapted from example code.
  
  Copyright ©2001-2004, Stewart Adcock <stewart@linux-domain.com>
  All rights reserved.

  The latest version of this program should be available at:
  http://gaul.sourceforge.net/

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.  Alternatively, if your project
  is incompatible with the GPL, I will probably agree to requests
  for permission to use the terms of any other license.

  This program is distributed in the hope that it will be useful, but
  WITHOUT ANY WARRANTY WHATSOEVER.

  A full copy of the GNU General Public License should be in the file
  "COPYING" provided with this distribution; if not, see:
  http://www.gnu.org/

 **********************************************************************/

/*
 * Includes
 */

#include "main_FOM_BANDGAP.cxx"
extern "C" {
    #include "gaul.h"
}



/**********************************************************************
  scoring and fitness functions - Jordan Walsh
 **********************************************************************/

// very simple fitness function, evaluates against a Gaussain centered at a target wavelength 
double gaussian_bg(double x) { return exp(-0.5 * pow((x - 1300) / 15, 2)); }
int count = 0;
static boolean score(population *pop, entity *entity)
{
	entity->fitness = 0.0;
	/* Extract QW characteristics from chromosomes of entity */
	std::vector<double> simulation_parameters;
	std::vector<Layer> input_layers;

	for(int i = 0; i < 3; i++){
		double x, y;
		GetXY( 0.8+(0.68675*(((double *)entity->chromosome[i])[1])), 0.0, x, y);
		if(count%20==0){
			std::cout << (((double *)entity->chromosome[i])[0]) << " " << (((double *)entity->chromosome[i])[1]) <<std::endl;
		}
		Material material(std::to_string(i), VB_InGaAlAs(x,y), CB_InGaAlAs(x,y), BG_InGaAlAs(x,y), effMass_InGaAlAs(x,y,0), effMass_InGaAlAs(x,y,1), effMass_InGaAlAs(x,y,2));
		input_layers.push_back(Layer(material,30.0+90.0*((double *)entity->chromosome[i])[0]));
	}
	
	if(count%20==0){
		std::cout << count << " structures simulated / fitness called." <<std::endl;
	}
	
	if(count == 0){ /*initial sanity check to make sure everything working */
		Heterostructure QW(input_layers); //works!
		QW.display(); //uncomment to see heterostructure information.
	}

	/* Evaluate entity */
	double bandgap = simulate(input_layers); // This is taking a LONG time
	entity->fitness = gaussian_bg(h_c/bandgap);
	
	count++;

	return TRUE;
}



/**********************************************************************
  main() - updated to double arrays
 **********************************************************************/

int main(int argc, char **argv)
{

	pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setstacksize(&attr, 16 * 1024 * 1024); // 16 MB allocated to stack
    
	omp_set_num_threads(4);
	Eigen::setNbThreads(4);
	Eigen::initParallel();
	
	
	population	*popd=NULL;		/* Population for evolution. */
	
	
	random_seed(2309012);
	


	popd = ga_genesis_double(
		25,									/* const int              population_size */
		3,										/* const int              num_chromo */
		(int)2,									/* const int              len_chromo */
		NULL,		 							/* GAgeneration_hook      generation_hook */
		NULL,									/* GAiteration_hook       iteration_hook */
		NULL,									/* GAdata_destructor      data_destructor */
		NULL,									/* GAdata_ref_incrementor data_ref_incrementor */ 
		score,									/* GAevaluate             evaluate */
		ga_seed_double_random,					/* GAseed                 seed */
		NULL,									/* GAadapt                adapt */
		ga_select_one_sus,						/* GAselect_one           select_one */
		ga_select_two_sus,						/* GAselect_two           select_two */
		ga_mutate_double_singlepoint_drift,		/* GAmutate    mutate */
		ga_crossover_double_mean,				/* GAcrossover         crossover */
		NULL,									/* GAreplace		replace */
		NULL									/* vpointer		User data */
			);

	//~ popd = ga_genesis_double( const int               population_size,
							//~ const int               num_chromo,
							//~ const int               len_chromo,
							//~ GAgeneration_hook       generation_hook,
							//~ GAiteration_hook        iteration_hook,
							//~ GAdata_destructor       data_destructor,
							//~ GAdata_ref_incrementor  data_ref_incrementor,
							//~ GAevaluate              evaluate,
							//~ GAseed                  seed,
							//~ GAadapt                 adapt,
							//~ GAselect_one            select_one,
							//~ GAselect_two            select_two,
							//~ GAmutate                mutate,
							//~ GAcrossover             crossover,
							//~ GAreplace               replace,
				//~ vpointer		userdata );

				
	ga_population_set_parameters(
		popd,			/* population   *pop */
		GA_SCHEME_DARWIN,		/* const ga_scheme_type scheme */
		GA_ELITISM_PARENTS_DIE,	/* const ga_elitism_type   elitism */
		0.9,			/* const double       crossover */
		0.1,			/* const double       mutation */
		0.0			/* const double       migration */
						);

	ga_population_set_allele_min_double( popd, 0.0);
	ga_population_set_allele_max_double( popd, 1.0);
	
/*
 * Evolve each population in turn.
 */

	ga_evolution(
		popd,			/* population          *pop */
		25				/* const int           max_generations */
			);
	
	std::cout << "fitness of best solution: " << ga_get_entity_from_rank(popd, 0)->fitness << std::endl;
	std::cout << "heterostructure info: " << std::endl;
	for (int i = 0; i<3; i++) {
		for (int j = 0; j < 2; j++) {
			std::cout << ((double *)ga_get_entity_from_rank(popd,0)->chromosome[i])[j] << " "; // lets just see what happens here..
		}
		std::cout << std::endl;
	}

  /* Deallocate population structures. */
	ga_extinction(popd);

	exit(EXIT_SUCCESS);
}


