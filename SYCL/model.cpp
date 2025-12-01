#include "model.h"
#include <cstdio>


uint64_t get_timestamp_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ((uint64_t)ts.tv_sec * 1000000000) + ts.tv_nsec;
}


void ReportProblemSizeCSV(const int sx, const int sy, const int sz,
			  const int bord, const int st,
			  FILE *f){
  fprintf(f,
	  "sx; %d; sy; %d; sz; %d; bord; %d;  st; %d; \n",
	  sx, sy, sz, bord, st);
}

void ReportMetricsCSV(double walltime, double MSamples,
		      long HWM, char *HWMUnit, FILE *f){
  fprintf(f,
	  "walltime; %lf; MSamples; %lf; HWM;  %ld; HWMUnit;  %s;\n",
	  walltime, MSamples, HWM, HWMUnit);
}

void Model(const int st, const int iSource, const float dtOutput, SlicePtr sPtr,
const int sx, const int sy, const int sz, const int bord,
const float dx, const float dy, const float dz, const float dt, const int it,
float* pp, float* pc, float* qp, float* qc,
float* vpz, float* vsv, float* epsilon, float* delta,
float* phi, float* theta, int absorb, sycl::queue q)
{
    //HOST SPACE VIEWS
    float* ch1dxx;  // isotropy simetry deep angle
    float* ch1dyy;  // isotropy simetry deep angle
    float* ch1dzz;  // isotropy simetry deep angle
    float* ch1dxy;  // isotropy simetry deep angle
    float* ch1dyz;  // isotropy simetry deep angle
    float* ch1dxz;  // isotropy simetry deep angle
    float* v2px;  // coeficient of H2(p)
    float* v2pz;  // coeficient of H1(q)
    float* v2sz;  // coeficient of H1(p-q) and H2(p-q)
    float* v2pn;  // coeficient of H2(p)

    //DEVICE SPACE VIEWS
    float* dev_ch1dxx;
    float* dev_ch1dyy;
    float* dev_ch1dzz;
    float* dev_ch1dxy;
    float* dev_ch1dyz;
    float* dev_ch1dxz;
    float* dev_v2px;
    float* dev_v2pz;
    float* dev_v2sz;
    float* dev_v2pn;
    float* dev_pp;
    float* dev_pc;
    float* dev_qp;
    float* dev_qc;

    float tSim=0.0;
    int nOut=1;
    float tOut=nOut*dtOutput;

    const long samplesPropagate=(long)(sx-2*bord)*(long)(sy-2*bord)*(long)(sz-2*bord);
    const long totalSamples=samplesPropagate*(long)st;

#ifdef PAPI
  long long values[NCOUNTERS];
  long long ThisValues[NCOUNTERS];
  for (int i=0; i<NCOUNTERS; i++) {
    values[i]=0LL;
    ThisValues[i]=0LL;
  }

  const int eventset=InitPAPI_CreateCounters();
#endif

printf("Iniit\n");
  // DRIVER_Initialize initialize target, allocate data etc
  DRIVER_Initialize(sx, sy, sz, bord,
            vpz, vsv, epsilon, delta, phi, theta,
	       ch1dxx, ch1dyy, ch1dzz,
	       ch1dxy, ch1dyz, ch1dxz,
	       v2px, v2pz, v2sz, v2pn,
		   dev_ch1dxx, dev_ch1dyy, dev_ch1dzz,
		   dev_ch1dxy, dev_ch1dyz, dev_ch1dxz,
		   dev_v2px, dev_v2pz, dev_v2sz, dev_v2pn,
		   dev_pp, dev_pc,
		   dev_qp, dev_qc, q);

  double walltime=0.0;
  double tdt=0.0;
  uint64_t stamp1 = get_timestamp_ns();

  int offset = 0;
  printf("Loop\n");

  for (int it=1; it<=st; it++) {

    // Calculate / obtain source value on i timestep
    float src = Source(dt, it-1);
    printf("Insert\n");

    DRIVER_InsertSource(dt,it-1,iSource,offset,dev_pc,dev_qc,src,q);

#ifdef PAPI
    StartCounters(eventset);
#endif

    const double t0=wtime();
printf("Propagate\n");
    DRIVER_Propagate(  sx,   sy,   sz,   bord,
		    dx,   dy,   dz,   dt,   it,
			offset,
		    pp,    pc,    qp,    qc,
			dev_ch1dxx, dev_ch1dyy, dev_ch1dzz,
			dev_ch1dxy, dev_ch1dyz, dev_ch1dxz,
			dev_v2px, dev_v2pz, dev_v2sz, dev_v2pn,
			dev_pp, dev_pc,
			dev_qp, dev_qc, q);

    SwapArrays(pp, pc, qp, qc);
    walltime+=wtime()-t0;

#ifdef PAPI
    StopReadCounters(eventset, ThisValues);
    for (int i=0; i<NCOUNTERS; i++) {
      values[i]+=ThisValues[i];
    }
#endif

    tSim=it*dt;
    if (tSim >= tOut) {
        printf("Update\n");

      DRIVER_Update_pointers(sx, sy, sz,pc, dev_pc, q);
      printf("Update ok\n");

      // double dd1 = wtime();
      // DumpSliceFile_Nofor(sx,sy,sz,pc,sPtr);
      // tdt+=wtime()-dd1;

      tOut=(++nOut)*dtOutput;
#ifdef _DUMP
      DRIVER_Update_pointers(sx,sy,sz,pc, dev_pc, q);
      //      DumpSliceSummary(sx,sy,sz,sPtr,dt,it,pc,src);
#endif
    }
  }

  // close binary output file before measuring time to include total io time
  CloseSliceFile(sPtr);

  uint64_t stamp2 = get_timestamp_ns();

  // get HWM data

#define MEGA 1.0e-6
#define GIGA 1.0e-9
  const char StringHWM[6]="VmHWM";
  char line[256], title[12],HWMUnit[8];
  long HWM;
  const double MSamples=(MEGA*(double)totalSamples)/walltime;

  FILE *fp=fopen("/proc/self/status","r");
  while (fgets(line, 256, fp) != NULL){
    if (strncmp(line, StringHWM, 5) == 0) {
      sscanf(line+6,"%ld %s", &HWM, HWMUnit);
      break;
    }
  }
  fclose(fp);

  // Dump Execution Metrics

	double execution_time = ((double)(stamp2-stamp1))*1e-9;

  // printf("Total dump time (s): %f\n", tdt);
  printf ("Execution time (s) is %lf\n", walltime);
  printf ("Total execution time (s) is %lf\n", execution_time);
  printf ("MSamples/s %.0lf\n", MSamples);
  printf ("Memory High Water Mark is %ld %s\n",HWM, HWMUnit);

  printf("original,%s,%d,%d,%d,%d,%.2f,%.2f,%.2f,%f,%f,%lu,%lu,%lf,%lf,%.0lf\n",
          sPtr->fName, sx - 2*bord - 2*absorb, sy - 2*bord - 2*absorb, sz - 2*bord - 2*absorb, absorb, dx, dy, dz, dt, st*dt,
          stamp1, stamp2, walltime, execution_time, MSamples);

  // Dump Execution Metrics in CSV

  FILE *fr=NULL;
  const char fName[]="Report.csv";
  fr=fopen(fName,"w");

  // report problem size

  ReportProblemSizeCSV(sx, sy, sz,
		       bord, st,
		       fr);

  // report collected metrics

  ReportMetricsCSV(walltime, MSamples,
		   HWM, HWMUnit, fr);

  // report PAPI metrics

#ifdef PAPI
  ReportRawCountersCSV (values, fr);
#endif

  fclose(fr);

  fflush(stdout);

  // DRIVER_Finalize deallocate data, clean-up things etc
  DRIVER_Finalize();

}
