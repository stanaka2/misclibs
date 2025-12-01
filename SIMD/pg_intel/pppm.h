#pragma once

#define SIZE_OF_MESH (1024.0)
#define CUTOFF_RAD_PM_MESH (3.0)
#define NPART_1D (256)

#undef SFT_FOR_PM
#undef SFT_FOR_PP

#define SFT_FOR_PM (CUTOFF_RAD_PM_MESH / SIZE_OF_MESH)
