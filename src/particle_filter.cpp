/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
    num_particles = 10;
    double std_x = std[0];
    double std_y = std[1];
    double std_theta = std[2];
    default_random_engine gen;
    
    normal_distribution<double> gaussianX(x, std_x);
    normal_distribution<double> gaussianY(y, std_y);
    normal_distribution<double> gaussianTheta(theta, std_theta);
    
    if (!is_initialized)
    {
        for (int i=0; i < num_particles; i++)
        {
            Particle tmp_particle;
            tmp_particle.id = i;
            tmp_particle.x = gaussianX(gen);
            tmp_particle.y = gaussianY(gen);
            tmp_particle.theta = gaussianTheta(gen);
            tmp_particle.weight = 1.0;
            
            particles.push_back(tmp_particle);
            weights.push_back(1.0);
            if (debug)
            {
                std::cout << i << " : x = " << tmp_particle.x << ", y = " << tmp_particle.y << ", theta = " << tmp_particle.theta << std::endl;
            }
        }
        is_initialized = true;
    }
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

    std::default_random_engine gen;
    
    for (int i=0; i < num_particles; i++)
    {
        double x, y, theta;
        if (fabs(yaw_rate < 1E-6))
        {
            x = particles.at(i).x + velocity*cos(particles.at(i).theta)*delta_t;
            y = particles.at(i).y + velocity*sin(particles.at(i).theta)*delta_t;
        }
        else
        {
            x = particles.at(i).x + velocity/yaw_rate*(sin(particles.at(i).theta + yaw_rate*delta_t) - sin(particles.at(i).theta));
            y = particles.at(i).y + velocity/yaw_rate*(cos(particles.at(i).theta) - cos(particles.at(i).theta + yaw_rate*delta_t));
        }
        theta = particles.at(i).theta + yaw_rate*delta_t;
        
        normal_distribution<double> gaussianX(x, std_pos[0]);
        normal_distribution<double> gaussianY(y, std_pos[1]);
        normal_distribution<double> gaussianTheta(theta, std_pos[2]);
        
        particles.at(i).x = gaussianX(gen);
        particles.at(i).y = gaussianY(gen);
        particles.at(i).theta = gaussianTheta(gen);
    }
    
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
    
    for (int i=0; i < observations.size(); i++)
    {
        int min_ind = -1;
        int minDist = 10000;
        for (int j=0; j < predicted.size(); j++)
        {
            double x_pred = predicted.at(j).x;
            double y_pred = predicted.at(j).y;
            double del_x = observations.at(i).x - x_pred;
            double del_y = observations.at(i).y - y_pred;
            
            double dist = sqrt(pow(del_x,2) + pow(del_y,2));
            
            if (dist < minDist)
            {
                min_ind = j;
                minDist = dist;
            }
        }
        observations.at(i).id = min_ind;
//        if (debug) {
//            std::cout << "Closest Landmark for Obs (" << observations.at(i).x << "," << observations.at(i).y << ")" << i << " is: " << predicted.at(min_ind).id << std::endl;
//        }
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

    // Transformation
    
    double sum_weights = 0.0;
    for (int i=0; i < particles.size(); i++)
    {
        std::vector<LandmarkObs> predicted;
        std::vector<LandmarkObs> observations_transformed;
        
        for (int j=0; j < observations.size(); j++)
        {
            LandmarkObs lObs;
            lObs.x = particles.at(i).x + cos(particles.at(i).theta)*observations.at(j).x - sin(particles.at(i).theta)*observations.at(j).y;
            lObs.y = particles.at(i).y + sin(particles.at(i).theta)*observations.at(j).x + cos(particles.at(i).theta)*observations.at(j).y;
            observations_transformed.push_back(lObs);
        }
        
        for (int j=0; j < map_landmarks.landmark_list.size(); j++)
        {
            double x_map = map_landmarks.landmark_list.at(j).x_f;
            double y_map = map_landmarks.landmark_list.at(j).y_f;
            if (fabs(x_map-particles.at(i).x) <= sensor_range && fabs(y_map-particles.at(i).y) <= sensor_range)
            {
                LandmarkObs lObs;
                lObs.id = map_landmarks.landmark_list.at(j).id_i;
                lObs.x = x_map;
                lObs.y = y_map;
                predicted.push_back(lObs);
            }
        }
        if (predicted.empty())
        {
            LandmarkObs lObs;
            lObs.x = 1E6;
            lObs.y = 1E6;
            predicted.push_back(lObs);
            std::cout << "No Map landmarks near current position" << std::endl;
        }
        dataAssociation(predicted, observations_transformed);
        double weight = 1.0;
        for (int j=0; j < observations_transformed.size(); j++)
        {
//            if (debug) {
//                std::cout << "Gaussian I/O for X: " << observations_transformed.at(j).id << "," << observations_transformed.at(j).x << ","  << predicted.at(observations_transformed.at(j).id).x << "," << std_landmark[0] << "," << normpdf(observations_transformed.at(j).x,predicted.at(observations_transformed.at(j).id).x,std_landmark[0]) << std::endl;
//                std::cout << "Gaussian I/O for Y: " << observations_transformed.at(j).id << "," << observations_transformed.at(j).y << ","  << predicted.at(observations_transformed.at(j).id).y << "," << std_landmark[1] << "," << normpdf(observations_transformed.at(j).y,predicted.at(observations_transformed.at(j).id).y,std_landmark[1]) << std::endl;
//            }
            weight *= normpdf(observations_transformed.at(j).x,predicted.at(observations_transformed.at(j).id).x,std_landmark[0]);
            weight *= normpdf(observations_transformed.at(j).y,predicted.at(observations_transformed.at(j).id).y,std_landmark[1]);
//            if (debug) {
//                std::cout << "Weight for obs(" << j << ") for particle " << i << " is " << weight << std::endl;
//            }
        }
        particles.at(i).weight = weight;
        weights.at(i) = weight;
        sum_weights += weight;
        
//        if (debug) {
//            std::cout << "Weight for particle " << i << " : " << weight << std::endl;
//        }
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
    default_random_engine gen;

    std::discrete_distribution<int> resampling_dist (weights.begin(),weights.end());
    
    std::vector<Particle> particles_resampled;
    for (int i=0; i < particles.size(); i++)
    {
        particles_resampled.push_back(particles.at(resampling_dist(gen)));
        particles_resampled.at(i).weight = 0.0;
    }
    particles = particles_resampled;
}

void ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
