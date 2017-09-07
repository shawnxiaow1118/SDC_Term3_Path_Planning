#include <fstream>
#include <math.h>
#include <uWS/uWS.h>
#include <iostream>
#include <vector>
#include <map>
#include <utility>
#include "Eigen-3.3/Eigen/Dense"
#include "json.hpp"
#include "spline.h"

#define MAP_FILE                "../data/highway_map.csv" // Original track

#define MAX_COST                   100000.0
#define PATH_PLAN_SECONDS		       2.5
#define PATH_PLAN_INCREMENT		     0.02
#define MAX_TRACK_S                6945.554

#define MIN_DIST              	   999999.0
#define MAX_V					             21.98
#define SLOW_DISTANCE			         30.0
#define SAFE_CHANGE				         18.0
#define LARGE   				           45.0
#define CHANGE_FRONT_SLOW          29.0
#define CHANGE_FRONT_FAST		       21.0
#define CHANGE_BEHIND_FAST		     30.0
#define CHANGE_BEHIND_SLOW		     23.0
#define CHANGE_GAP				         1.0

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;
using std::map;
using json = nlohmann::json;


constexpr double pi()    { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }


string hasData(string s) 
{
    auto found_null = s.find("null");
    auto b1         = s.find_first_of("[");
    auto b2         = s.find_first_of("}");
    if (found_null != string::npos) 
    {
        return "";
    } 
    else if (b1 != string::npos && b2 != string::npos) 
    {
        return s.substr(b1, b2 - b1 + 2);
    }
    return "";
}

double distance(double x1, double y1, double x2, double y2)
{
    return sqrt((x2-x1) * (x2-x1) + (y2-y1) * (y2-y1));
}

// calculate the cloeset waypoint
int ClosestWaypoint(double x, double y, vector<double> maps_x, vector<double> maps_y)
{
    double closestLen   = 100000.0; 
    int closestWaypoint = 0;
    for (int i=0; i<maps_x.size(); i++)
    {
        double map_x = maps_x[i];
        double map_y = maps_y[i];
        double dist  = distance(x, y, map_x, map_y);
        if (dist < closestLen)
        {
            closestLen      = dist;
            closestWaypoint = i;
        }
    }
    return closestWaypoint;
}

// calculate next waypoint given current location and direction
int NextWaypoint(double x, double y, double theta, vector<double> maps_x, vector<double> maps_y, vector<double> maps_dx, vector<double> maps_dy)
{

	int closestWaypoint = ClosestWaypoint(x,y,maps_x,maps_y);

	double map_x = maps_x[closestWaypoint];
	double map_y = maps_y[closestWaypoint];

    double hx = map_x-x;
    double hy = map_y-y;
    
    double nx = maps_dx[closestWaypoint];
    double ny = maps_dy[closestWaypoint];
    
    double vx = -ny;
    double vy = nx;

    double inner = hx*vx+hy*vy;
    if (inner<0.0) {
        closestWaypoint++;
    }

    return closestWaypoint;
}

// Transform from cartesian x,y coordinates to frenet s,d coordinates
vector<double> getFrenet(double x, double y, double theta, vector<double> maps_x, vector<double> maps_y,vector<double> maps_dx, vector<double> maps_dy)
{
	int next_wp = NextWaypoint(x,y, theta, maps_x,maps_y, maps_dx, maps_dy);

	int prev_wp;
	prev_wp = next_wp-1;
	if(next_wp == 0)
	{
		prev_wp  = maps_x.size()-1;
	}

	double n_x = maps_x[next_wp]-maps_x[prev_wp];
	double n_y = maps_y[next_wp]-maps_y[prev_wp];
	double x_x = x - maps_x[prev_wp];
	double x_y = y - maps_y[prev_wp];

	double proj_norm = (x_x*n_x+x_y*n_y)/(n_x*n_x+n_y*n_y);
	double proj_x = proj_norm*n_x;
	double proj_y = proj_norm*n_y;

	double frenet_d = distance(x_x,x_y,proj_x,proj_y);

	double center_x = 1000-maps_x[prev_wp];
	double center_y = 2000-maps_y[prev_wp];
	double centerToPos = distance(center_x,center_y,x_x,x_y);
	double centerToRef = distance(center_x,center_y,proj_x,proj_y);

	if(centerToPos <= centerToRef)
	{
		frenet_d *= -1;
	}

	double frenet_s = 0;
	for(int i = 0; i < prev_wp; i++)
	{
		frenet_s += distance(maps_x[i],maps_y[i],maps_x[i+1],maps_y[i+1]);
	}

	frenet_s += distance(0,0,proj_x,proj_y);

	return {frenet_s,frenet_d};

}

// Transfer from frenet s,d coordinates to cartesian x,y coordinates
vector<double> getXY(double s, double d, vector<double> maps_s, vector<double> maps_x, vector<double> maps_y)
{
    int prev_wp = -1;
    while (s > maps_s[prev_wp+1] && (prev_wp < (int)(maps_s.size()-1)))
    {
        prev_wp++;
    }
    int wp2             = (prev_wp+1) % maps_x.size();
    double heading      = atan2((maps_y[wp2] - maps_y[prev_wp]), (maps_x[wp2] - maps_x[prev_wp]));
    double seg_s        = s - maps_s[prev_wp];
    double seg_x        = maps_x[prev_wp] + seg_s * cos(heading);
    double seg_y        = maps_y[prev_wp] + seg_s * sin(heading);
    double perp_heading = heading - pi() / 2;
    double x            = seg_x + d * cos(perp_heading);
    double y            = seg_y + d * sin(perp_heading);
    return {x, y};
}

// car struat
struct car
{
	int id;
  	double car_s;
  	double car_d;
  	double car_v;
  	double dist;
};

struct save_state_t
{
    double last_s;
    double last_d;
    double last_speed;
};

// ego car struct
struct ego_car
{
  	double car_s;
  	double car_d;
  	double car_v;
  	vector<car> other_cars;
};

struct state
{
  	double end_s;
  	double end_d;
  	double end_v;
};

struct Action
{
  	int lane;
  	double ref_vel;
  	int change_wp;
};

struct path
{
    vector<double> path_x;
    vector<double> path_y;
    double last_s;
    double last_d;
};

double lane_to_d(int lane)
{
    return 2.0+4.0*lane;
}

int d_to_lane(double d)
{
  	if (d<4.0)
  	{
    	return 0;
  	}
  	if (d>=4.0 && d<8.0)
  	{
    	return 1;
  	}
  	if (d>=8.0)
  	{
     	return 2;
  	}
  	return -1;
}

// closest car in front given a lane and sensor fusion
car closest_front_car(ego_car my_car, int lane)
{
  	double min_dist = MIN_DIST;
  	double min_id = -1;
  	for (int i = 0; i < my_car.other_cars.size(); i++)
  	{
    	if (d_to_lane(my_car.other_cars[i].car_d) == lane)
    	{
      		double cur_dist = my_car.other_cars[i].car_s - my_car.car_s;
      		my_car.other_cars[i].dist = cur_dist;
      		if (cur_dist >= 0.0 && cur_dist < min_dist)
      		{
        		min_dist = cur_dist;
        		min_id = i;
      		}
    	}
  	}

  	if (min_id == -1)
  	{
    	car dummy_car = {-1, 0.0,0.0,0.0, MIN_DIST};
    	return dummy_car;
  	}
  	return my_car.other_cars[min_id];
}

// closest car behind given a lane and sensor fusion
car closest_behind_car(ego_car my_car, int lane)
{
  	double min_dist = MIN_DIST;
  	double min_id = -1;
  	for (int i = 0; i < my_car.other_cars.size(); i++)
  	{
    	if (d_to_lane(my_car.other_cars[i].car_d) == lane)
    	{
     		double cur_dist = my_car.car_s - my_car.other_cars[i].car_s;
      		my_car.other_cars[i].dist = cur_dist;
      		if (cur_dist >= 0.0 && cur_dist < min_dist)
      		{
        		min_dist = cur_dist;
        		min_id = i;
      		}
    	}
  	}

  	if (min_id == -1)
  	{
    	car dummy_car = {-1, 0,0,0, MIN_DIST};
    	return dummy_car;
  	}
  	return my_car.other_cars[min_id];
}

// calculate the most efficient behavior 
Action choose_action(ego_car my_car, int change_wp, int next_wp)
{
	// ofstream myfile;
	// myfile.open ("example.txt", std::ofstream::out | std::ofstream::app);
  	int lane = d_to_lane(my_car.car_d);
  	double ref_vel = my_car.car_v;
  	int action = -1;
  	// cout << " buffer: " << abs(next_wp - change_wp) << endl;
  	// myfile <<  "buffer: " << abs(next_wp - change_wp) << "\n";
  	bool try_change_lane = false;
  	bool change_left = false;
  	bool change_right = false;

  	car front_car = closest_front_car(my_car, lane);
  	car behind_car = closest_behind_car(my_car, lane);
  	car left_front_car = closest_front_car(my_car, lane-1);
  	car left_behind_car = closest_behind_car(my_car, lane-1);
  	car right_front_car = closest_front_car(my_car, lane+1);
  	car right_behind_car = closest_behind_car(my_car, lane+1);

  	bool left_acc = false;
  	bool right_acc = false;

  	// no car in front or front car 
  	if (front_car.id==-1 || (front_car.id!= -1 && front_car.car_v > my_car.car_v && front_car.car_v >= MAX_V) || front_car.dist > SLOW_DISTANCE)
  	{
    	action = 0;
    	ref_vel = MAX_V;
    	Action act = {lane, ref_vel, change_wp};
    	return act;
  	}
  	else if (front_car.dist < SLOW_DISTANCE)
  	{
    	try_change_lane = true;
  	}


  	if (try_change_lane && abs(next_wp - change_wp) > 2)
  	{
    	// turn left
    	if (lane == 1 || lane == 2)
    	{
    		// myfile << "left dist: "<<left_front_car.dist << " left_behind dist: " << left_behind_car.dist << "\n";
      		if (left_front_car.dist < SAFE_CHANGE || left_behind_car.dist < SAFE_CHANGE)
      		{
        		change_left = false;
      		}
      		else
      		{
      			// myfile << "left front id:" <<left_front_car.id << " behind_id " << left_behind_car.id << "\n";
      			// myfile << "left front v: " << left_front_car.car_v << " vb: " << left_behind_car.car_v << "\n";
    			if (left_front_car.dist >= LARGE && left_behind_car.dist >= LARGE)
    			{
    				change_left = true;
    				left_acc = true;
    			}
    			else if (left_front_car.dist >= LARGE && left_behind_car.dist < LARGE)
    			{
    	  			if (left_behind_car.car_v > my_car.car_v && left_behind_car.car_v < MAX_V)
          			{
            			if (left_behind_car.dist >= CHANGE_BEHIND_FAST)
            			{
              				change_left = true;
              				left_acc = true;
            			}
          			} 
          			else
          			{
            			if (left_behind_car.dist > CHANGE_BEHIND_SLOW)
            			{
              				change_left = true;
            			}
          			}
    			}
    			else if (left_behind_car.dist >= LARGE && left_front_car.dist < LARGE)
    			{
          			if (left_front_car.car_v > front_car.car_v + CHANGE_GAP)
          			{
            			if (left_front_car.car_v >= my_car.car_v && left_front_car.dist >= CHANGE_FRONT_FAST)
            			{
              				change_left = true;
            			}
            			else if(left_front_car.dist >= CHANGE_FRONT_SLOW)
            			{
              				change_left = true;
            			}
          			}       		
    			}
    			else
    			{
    				if (left_front_car.car_v > front_car.car_v + CHANGE_GAP)
    				{
    					if (left_front_car.car_v >= my_car.car_v && left_front_car.dist >= CHANGE_FRONT_FAST)
    					{
    						if (left_behind_car.car_v <= my_car.car_v)
    						{
    							change_left = true;
    						} 
    						else
    						{
    							if (left_behind_car.dist >= CHANGE_BEHIND_FAST)
    							{
    								change_left = true;
    							}
    						}
    					}
    					else
    					{
    						if (left_front_car.dist >= CHANGE_FRONT_SLOW)
    						{
    							if (left_behind_car.car_v < my_car.car_v)
    							{
    								change_left = true;
    							}
    							else
    							{
    								if (left_behind_car.dist >= CHANGE_BEHIND_FAST)
    								{
    									change_left = true;
    								}
    							}
    						}
    					}
    				}
    			}
      		}
    	}

    	//turn right
   		if (lane == 0 || lane == 1)
    	{
    		// myfile << "right dist: "<<right_front_car.dist << " right_behind dist: " << right_behind_car.dist << "\n";
      		if (right_front_car.dist < SAFE_CHANGE || right_behind_car.dist < SAFE_CHANGE)
      		{
        		change_right = false;
      		}
      		else
      		{
      	 		// myfile << " right front id:" <<right_front_car.id << " right behind_id " << right_behind_car.id << "\n";
      			// myfile << "right front v: " << right_front_car.car_v << " right vb: " << right_behind_car.car_v << "\n";
    			if (right_front_car.dist >= LARGE && right_behind_car.dist >= LARGE)
    			{
    				change_right = true;
    				right_acc = true;
    			}
    			else if (right_front_car.dist >= LARGE && right_behind_car.dist < LARGE)
    			{
    	  			if (right_behind_car.car_v > my_car.car_v && right_behind_car.car_v < MAX_V)
          			{
            			if (right_behind_car.dist >= CHANGE_BEHIND_FAST)
            			{
              				change_right = true;
              				right_acc = true;
            			}
          			} 
          			else
          			{
            			if (right_behind_car.dist > CHANGE_BEHIND_SLOW)
            			{
              				change_right = true;
            			}
          			}
    			}
    			else if (right_behind_car.dist >= LARGE && right_front_car.dist < LARGE)
    			{
          			if (right_front_car.car_v > front_car.car_v + CHANGE_GAP)
          			{
            			if (right_front_car.car_v >= my_car.car_v && right_front_car.dist >= CHANGE_FRONT_FAST)
            			{
              				change_right = true;
            			}
            			else if(right_front_car.dist >= CHANGE_FRONT_SLOW)
            			{
              				change_right = true;
            			}
          			}       		
    			}
    			else
    			{
    				if (right_front_car.car_v > front_car.car_v + CHANGE_GAP)
    				{
    					if (right_front_car.car_v >= my_car.car_v && right_front_car.dist >= CHANGE_FRONT_FAST)
    					{
    						if (right_behind_car.car_v <= my_car.car_v)
    						{
    							change_right = true;
    						} 
    						else
    						{
    							if (right_behind_car.dist >= CHANGE_BEHIND_FAST)
    							{
    								change_right = true;
    							}
    						}
    					}
    					else
    					{
    						if (right_front_car.dist >= CHANGE_FRONT_SLOW)
    						{
    							if (right_behind_car.car_v < my_car.car_v)
    							{
    								change_right = true;
    							}
    							else
    							{
    								if (right_behind_car.dist >= CHANGE_BEHIND_FAST)
    								{
    									change_right = true;
    								}
    							}
    						}
    					}
    				}
    			}
      		}
    	}
 	}

  	// the lane to the other side, avoid change to the same lane at the same time
  	if (lane == 0)
  	{
  		for (int i = 0; i < my_car.other_cars.size(); i++)
  		{
  			if (d_to_lane(my_car.other_cars[i].car_d) == 2)
  			{
  				if (abs(my_car.other_cars[i].car_s - my_car.car_s) < 4.0)
  				{
  					change_left = false;
  					change_right = false;
  				}
  			}
  		}
  	}

  	if (lane == 2)
  	{
  		for (int i = 0; i < my_car.other_cars.size(); i++)
  		{
  			if (d_to_lane(my_car.other_cars[i].car_d) == 0)
  			{
  				if (abs(my_car.other_cars[i].car_s - my_car.car_s) < 10.0)
  				{
  					change_left = false;
  					change_right = false;
  				}
  			}
  		}
  	}

  	// get the safe and fast velocity and lane
  	if (change_left && change_right)
  	{
  		if (left_front_car.dist > LARGE)
  		{
  			action = -1;
  		}
  		else if (right_front_car.dist > LARGE)
  		{
  			action = 1;
  		}
  		else
  		{
  			if (left_front_car.car_v > right_front_car.car_v)
  			{
  				action = -1;
  			}	 
  			else
  			{
  				action = 1;
  			}
  		}
  		if (action == 1)
  		{
	    	if (right_acc)
	    	{
	    		if (right_front_car.dist > LARGE)
	    		{
	    			ref_vel = MAX_V;
	    		} 
	    		else
	    		{
	    			ref_vel = min(MAX_V, right_front_car.car_v);
	    		}
	    	}
  		} 
  		else
  		{
	    	if (left_acc)
	    	{
	    		if (left_front_car.dist > LARGE)
	    		{
	    			ref_vel = MAX_V;
	    		} 
	    		else
	    		{
	    			ref_vel = min(MAX_V, left_front_car.car_v);
	    		}
	    	}
  		}
  	}
  	else if(!change_left && change_right)
  	{
    	action = 1;
    	if (right_acc)
    	{
    		if (right_front_car.dist > LARGE)
    		{
    			ref_vel = MAX_V;
    		} 
    		else
    		{
    			ref_vel = min(MAX_V, right_front_car.car_v);
    		}
    	}
  	}
  	else if(change_left && !change_right)
  	{
   		action = -1;
    	if (left_acc)
    	{
    		if (left_front_car.dist > LARGE)
    		{
    			ref_vel = MAX_V;
    		} 
    		else
    		{
    			ref_vel = min(MAX_V, left_front_car.car_v);
    		}
    	}
  	}
  	else
  	{
    	action = 0;
    	if (front_car.dist > 20.0)
    	{
      		ref_vel = min(front_car.car_v, MAX_V);
    	} 
    	else
    	{
      		ref_vel = front_car.car_v - 2.3;
    	}
  	}
  	int new_lane = lane;
  	if (action == -1) 
  	{
  		new_lane = lane-1;
  		change_wp = next_wp;
  	}
  	else if (action == 0)
  	{
  		new_lane = lane;
  	}
  	else
  	{
  		new_lane = lane + 1;
  		change_wp = next_wp;
  	}

  	Action act = {new_lane, ref_vel, change_wp};
  	return act;
}

// Generate targets if change to left lane
state generate_left_target(ego_car my_car, double ref_vel)
{
	double s = 0;
	double speed_start = my_car.car_v;
	double speed_end = ref_vel;

	state end_state = {
		my_car.car_s + PATH_PLAN_SECONDS * 0.5 * (speed_start + speed_end),
		lane_to_d(d_to_lane(my_car.car_d)-1),
		speed_end
	};
	return end_state;
}

// Generate targets if change to right lane
state generate_right_target(ego_car my_car, double ref_vel)
{
	double s = 0;
	double speed_start = my_car.car_v;
	double speed_end = ref_vel;

	state end_state = {
		my_car.car_s + PATH_PLAN_SECONDS * 0.5 * (speed_start + speed_end),
		lane_to_d(d_to_lane(my_car.car_d)+1),
		speed_end
	};
	return end_state;
}

// Generate targets if stay in the same line
state generate_lane_target(ego_car my_car, double ref_vel)
{
	int lane = d_to_lane(my_car.car_d);
	double speed_start = my_car.car_v;
	double speed_end = ref_vel;

    cout << "speed_start " << speed_start << endl;
    cout << "speed_end " << speed_end << endl << endl;

    state end_state = {
        my_car.car_s + PATH_PLAN_SECONDS * 0.5 * (speed_start + speed_end),
        lane_to_d(lane),
        speed_end
    };
    return end_state;
}

// Genereate jerk minimiazation path, not used in the final version
// path JMT_Path(state start_state, state end_state, vector<double> map_waypoints_s,vector<double> map_waypoints_x,vector<double> map_waypoints_y)
// {
//     // Conditions for minimum jerk in s (zero start/end acceleration) 
//     double start_pos_s = start_state.end_s;
//     double start_vel_s = start_state.end_v; 
//     double end_pos_s   = end_state.end_s; 
//     double end_vel_s   = end_state.end_v; 

//     // Conditions for minimum jerk in d (zero start/end acceleration and velocity, indexing by lane) 
//     double start_pos_d = start_state.end_d;
//     double end_pos_d   = end_state.end_d;
//     // Generate minimum jerk path in Frenet coordinates
//     vector<double> next_s_vals = computeMinimumJerk({start_pos_s, start_vel_s, 0.0}, {end_pos_s,   end_vel_s,   0.0}, 
//                                                     PATH_PLAN_SECONDS, PATH_PLAN_INCREMENT);
//     vector<double> next_d_vals = computeMinimumJerk({start_pos_d, 0.0, 0.0}, {end_pos_d,   0.0, 0.0}, 
//                                                     PATH_PLAN_SECONDS,PATH_PLAN_INCREMENT);

//     // Convert Frenet coordinates to map coordinates
//     vector<double> next_x_vals = {};
//     vector<double> next_y_vals = {};
//     for (int i=0; i<next_s_vals.size(); i++)
//     {
//     	// cout << next_s_vals[i] << " " << next_d_vals[i] << endl;
//         vector<double> xy = getXY(fmod(next_s_vals[i], MAX_TRACK_S),
//                                   next_d_vals[i],
//                                   map_waypoints_s,
//                                   map_waypoints_x,
//                                   map_waypoints_y);
//         next_x_vals.push_back(xy[0]);
//         next_y_vals.push_back(xy[1]);
//     }
//     path result;
//     result.path_x = next_x_vals;
//     result.path_y = next_y_vals;
//     result.last_s = next_s_vals[next_s_vals.size()-1];
//     result.last_d = next_d_vals[next_d_vals.size()-1];
//     return result;
// }



int main() {
  uWS::Hub h;

  vector<double> map_waypoints_x;
  vector<double> map_waypoints_y;
  vector<double> map_waypoints_s;
  vector<double> map_waypoints_dx;
  vector<double> map_waypoints_dy;

  string map_file_ = MAP_FILE;
  double max_s = 6945.554;

  ifstream in_map_(map_file_.c_str(), ifstream::in);

  string line;
  while (getline(in_map_, line)) {
  	istringstream iss(line);
  	double x;
  	double y;
  	float s;
  	float d_x;
  	float d_y;
  	iss >> x;
  	iss >> y;
  	iss >> s;
  	iss >> d_x;
  	iss >> d_y;
  	map_waypoints_x.push_back(x);
  	map_waypoints_y.push_back(y);
  	map_waypoints_s.push_back(s);
  	map_waypoints_dx.push_back(d_x);
  	map_waypoints_dy.push_back(d_y);
  }

  //set initial lane
  int lane = 1;
  // flag to avoid car change line too often
  int change_wp = 0;

  h.onMessage([&map_waypoints_x,&map_waypoints_y,&map_waypoints_s,&map_waypoints_dx,&map_waypoints_dy,&lane,&change_wp]
  	(uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    //auto sdata = string(data).substr(0, length);
    //cout << sdata << endl;
    if (length && length > 2 && data[0] == '4' && data[1] == '2') {

      auto s = hasData(data);

      if (s != "") {
        auto j = json::parse(s); 
        string event = j[0].get<string>();
        
        if (event == "telemetry") {
          	double car_x = j[1]["x"];
          	double car_y = j[1]["y"];
          	double car_s = j[1]["s"];
          	double car_d = j[1]["d"];
          	double car_yaw = j[1]["yaw"];
          	double car_speed = j[1]["speed"];
          	auto previous_path_x = j[1]["previous_path_x"];
          	auto previous_path_y = j[1]["previous_path_y"];
          	double end_path_s = j[1]["end_path_s"];
          	double end_path_d = j[1]["end_path_d"];
          	vector<vector<double>> sensor_fusion = j[1]["sensor_fusion"];

          	double target_vel = 49.3;

          	int next_wp = -1;
          	double ref_x = car_x;
          	double ref_y = car_y;
          	double ref_yaw = deg2rad(car_yaw);

          	// starting run, no previous path
          	if(previous_path_x.size() < 2)
          	{
          		next_wp = NextWaypoint(ref_x, ref_y, ref_yaw, map_waypoints_x,map_waypoints_y,map_waypoints_dx,map_waypoints_dy);
          		// for some map, have the determine the start lane and waypoint
          		change_wp = next_wp;
          		lane = d_to_lane(car_d);
          	}
          	else
          	{
      				ref_x = previous_path_x[previous_path_x.size()-1];
      				double ref_x_prev = previous_path_x[previous_path_x.size()-2];
      				ref_y = previous_path_y[previous_path_x.size()-1];
      				double ref_y_prev = previous_path_y[previous_path_x.size()-2];
      				ref_yaw = atan2(ref_y-ref_y_prev,ref_x-ref_x_prev);
      				next_wp = NextWaypoint(ref_x,ref_y,ref_yaw,map_waypoints_x,map_waypoints_y,map_waypoints_dx,map_waypoints_dy);
      				// get the car information
      				car_s = end_path_s;
              car_d = end_path_d;
      				car_speed = (sqrt((ref_x-ref_x_prev)*(ref_x-ref_x_prev)+(ref_y-ref_y_prev)*(ref_y-ref_y_prev))/.02)*2.237;
          	}

  	        vector<car> other_cars = {};
  	        for (int i = 0; i < sensor_fusion.size(); i++) 
  	        {
            	int id     	    = sensor_fusion[i][0];
            	double s       = sensor_fusion[i][5];
            	double d       = sensor_fusion[i][6];
            	double vx      = sensor_fusion[i][3];
            	double vy      = sensor_fusion[i][4];
            	double v   	 = sqrt(vx*vx + vy*vy);
            	s += v*0.02*(double)previous_path_x.size();
            	car new_car = {id,s,d,v,MIN_DIST};
            	other_cars.push_back(new_car);
  	        }
  	        ego_car my_car = {car_s, lane_to_d(lane), car_speed/2.237, other_cars};

	        // Get the best act
          	Action act = choose_action(my_car, change_wp, next_wp);
          	lane = act.lane;
          	// Transfer to lane and velocity information
          	target_vel = act.ref_vel*2.237;
          	change_wp = act.change_wp;          	

          	vector<double> ptsx;
          	vector<double> ptsy;
          	vector<double> next_x_vals;
          	vector<double> next_y_vals;

          	if(previous_path_x.size() < 2)
          	{
          		double car_prev_x = car_x - cos(car_yaw);
          		double car_prev_y = car_y - sin(car_yaw);

          		ptsx.push_back(car_prev_x);
          		ptsx.push_back(car_x);
          		ptsy.push_back(car_prev_y);
          		ptsy.push_back(car_y);
          	}
          	else
          	{
          		ptsx.push_back(previous_path_x[previous_path_x.size()-2]);
          		ptsx.push_back(previous_path_x[previous_path_x.size()-1]);

          		ptsy.push_back(previous_path_y[previous_path_x.size()-2]);
          		ptsy.push_back(previous_path_y[previous_path_x.size()-1]);
          	}
            // points for spline to use, contains 2 current points and 3 waypoints after
          	vector<double> interpolated_0 = getXY(car_s+30,(2+4*lane),map_waypoints_s,map_waypoints_x,map_waypoints_y);
          	vector<double> interpolated_1 = getXY(car_s+60,(2+4*lane),map_waypoints_s,map_waypoints_x,map_waypoints_y);
          	vector<double> interpolated_2 = getXY(car_s+90,(2+4*lane),map_waypoints_s,map_waypoints_x,map_waypoints_y);

          	ptsx.push_back(interpolated_0[0]);
          	ptsx.push_back(interpolated_1[0]);
          	ptsx.push_back(interpolated_2[0]);
          	ptsy.push_back(interpolated_0[1]);
          	ptsy.push_back(interpolated_1[1]);
          	ptsy.push_back(interpolated_2[1]);

          	// transfer to car centered relative coordinate system
          	for (int i = 0; i < ptsx.size(); i++ )
          	{
          		double shift_x = ptsx[i]-ref_x;
          		double shift_y = ptsy[i]-ref_y;
      				ptsx[i] = (shift_x *cos(-ref_yaw)-shift_y*sin(-ref_yaw));
      				ptsy[i] = (shift_x *sin(-ref_yaw)+shift_y*cos(-ref_yaw));
          	}    	
          	// spline to interpolated the waypoints
          	tk::spline s;
          	s.set_points(ptsx,ptsy);

          	for(int i = 0; i < previous_path_x.size(); i++)
          	{
          		next_x_vals.push_back(previous_path_x[i]);
          		next_y_vals.push_back(previous_path_y[i]);
          	}
          	// in car centered coordinate system, want the car to move in x axis for 30.0 meters
          	double target_x = 30.0;
          	double target_y = s(target_x);
          	double target_dist = sqrt((target_x)*(target_x)+(target_y)*(target_y));
          	
          	double x_add_on = 0;
          	// generate new path point
      			for (int i = 1; i <= 50-previous_path_x.size(); i++) {
      				
      				if(target_vel > car_speed)
      				{
      					car_speed+=.300;
      				}
      				else if(target_vel < car_speed)
      				{
      					car_speed-=.200;
      				}

      				double N = (target_dist/(.02*car_speed/2.237));
      				double x_point = x_add_on+(target_x)/N;
      				double y_point = s(x_point);
      				x_add_on = x_point;
      				// transfer back to global cartisian coordiante system
      				double x_ref = x_point;
      				double y_ref = y_point;
      				x_point = (x_ref *cos(ref_yaw)-y_ref*sin(ref_yaw));
      				y_point = (x_ref *sin(ref_yaw)+y_ref*cos(ref_yaw));
      				x_point += ref_x;
      				y_point += ref_y;

      				next_x_vals.push_back(x_point);
      				next_y_vals.push_back(y_point);
      			}
          	json msgJson;
          	msgJson["next_x"] = next_x_vals;
          	msgJson["next_y"] = next_y_vals;

          	auto msg = "42[\"control\","+ msgJson.dump()+"]";

          	//this_thread::sleep_for(chrono::milliseconds(1000));
          	ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
          
        }
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }
  });

  // We don't need this since we're not using HTTP but if it's removed the
  // program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
                     size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    if (req.getUrl().valueLength == 1) {
      res->end(s.data(), s.length());
    } else {
      // i guess this should be done more gracefully?
      res->end(nullptr, 0);
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}
