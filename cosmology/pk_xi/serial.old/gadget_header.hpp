#pragma once

namespace gadget
{
struct header {
  int npart[6];    //
  double mass[6];  //
  double time;     //
  double redshift; //
  int flag_sfr;
  int flag_feedback;
  unsigned int npartTotal[6]; //
  int flag_cooling;
  int num_files;      //
  double BoxSize;     //
  double Omega0;      //
  double OmegaLambda; //
  double HubbleParam; //
  int flag_stellarage;
  int flag_metals;
  int hashsize;
  float disp_min;
  float disp_max;
  int pos_bits;
  long long int id_start;
  long long int id_end;
  int output_pot;
  char fill[52];
};

} // namespace gadget
