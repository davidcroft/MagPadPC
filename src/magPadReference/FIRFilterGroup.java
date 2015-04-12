package magPadReference;

public class FIRFilterGroup {
	// FirFilter
	public FirFilter firFilterLowPassX, firFilterLowPassY, firFilterLowPassZ; // low pass filter
	
	public FIRFilterGroup() {
		// low pass filter
		firFilterLowPassX = new FirFilter("lowPass.fcf");
		firFilterLowPassY = new FirFilter("lowPass.fcf");
		firFilterLowPassZ = new FirFilter("lowPass.fcf");
	}
	
	public float applyToFilter(float val, int axisIndex) {
		switch (axisIndex) {
			case 0: return (float)firFilterLowPassX.filter((double)val);
			case 1: return (float)firFilterLowPassY.filter((double)val);
			case 2: return (float)firFilterLowPassZ.filter((double)val);
			default: return 0;
		}
	}
}
