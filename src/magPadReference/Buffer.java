package magPadReference;

public class Buffer {
	  
	public static int BUFFERSIZE_t = GlobalConstants.BUFFERSIZE;
	public static int BUFFERNUM = 128/GlobalConstants.BUFFERSIZE;
	
	// val is used to save the three 
	public float[] m_bufX;
	public float[] m_bufY;
	public float[] m_bufZ;
	
	private int m_bufferSegSize;
	// m_bufferIndex is used to identify which buffer is used to storage
	public int m_bufferIndex;
	
	public Buffer(int bufferSize) {
		// init a new buffer with 4 subsegments
	    m_bufX = new float[bufferSize*BUFFERNUM];
	    m_bufY = new float[bufferSize*BUFFERNUM];
	    m_bufZ = new float[bufferSize*BUFFERNUM];
	    
	    m_bufferSegSize = bufferSize;
	    m_bufferIndex = 0;
	}
	    
	public Buffer() {
	    m_bufX = new float[BUFFERSIZE_t*BUFFERNUM];
	    m_bufY = new float[BUFFERSIZE_t*BUFFERNUM];
	    m_bufZ = new float[BUFFERSIZE_t*BUFFERNUM];
	    
	    m_bufferSegSize = BUFFERSIZE_t;
	    m_bufferIndex = 0;  
	}

	public void addToBuffer(float[] xBufSeg, float[] yBufSeg, float[] zBufSeg) {
		// storage vals
	    for (int i = 0; i < m_bufferSegSize; i++) {
	    	int index = m_bufferIndex*m_bufferSegSize + i;
	    	// save to buffer
	    	m_bufX[index] = xBufSeg[i];
	    	m_bufY[index] = yBufSeg[i];
	    	m_bufZ[index] = zBufSeg[i];
	    }
	    updateIndex();
	}
	  
	public void updateIndex() {
		m_bufferIndex = (++m_bufferIndex)%BUFFERNUM;
	    //System.out.println("m_bufferIndex = " + m_bufferIndex);
	}
	  
	public float[] genBufferForFFT(int startIndex, int axis) {
	    float[] result = new float[m_bufferSegSize*BUFFERNUM];
	    int cnt = 0;
	    if (startIndex < 0) startIndex += BUFFERNUM;
	    int index = startIndex*m_bufferSegSize;
	    if (axis == 1) {
	    	// x axis
	    	for(int i = index; i < m_bufferSegSize*BUFFERNUM; i++) {
	    		result[cnt++] = m_bufX[i];
	    	}
	    	for (int i = 0; i < startIndex*m_bufferSegSize; i++) {
	    		result[cnt++] = m_bufX[i];
	    	}
	    } else if (axis == 2) {
	    	// y axis
	    	for(int i = index; i < m_bufferSegSize*BUFFERNUM; i++) {
	    		result[cnt++] = m_bufY[i];
	    	}
	    	for (int i = 0; i < startIndex*m_bufferSegSize; i++) {
	    		result[cnt++] = m_bufY[i];
	    	}
	    } else if (axis == 3) {
	    	for(int i = index; i < m_bufferSegSize*BUFFERNUM; i++) {
	    		result[cnt++] = m_bufZ[i];
	    	}
	    	for (int i = 0; i < startIndex*m_bufferSegSize; i++) {
	    		result[cnt++] = m_bufZ[i];
	    	}
	    }
	    return result;
	}

}