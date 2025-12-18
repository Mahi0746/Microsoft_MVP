import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import { useAuth } from '../contexts/AuthContext';
import DashboardLayout from '../components/layout/DashboardLayout';
import LoadingSpinner from '../components/ui/LoadingSpinner';

const ARScannerPage: React.FC = () => {
  const { user, loading } = useAuth();
  const router = useRouter();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [documentType, setDocumentType] = useState('prescription');
  const [isScanning, setIsScanning] = useState(false);
  const [scanResult, setScanResult] = useState<any>(null);
  const [recentScans, setRecentScans] = useState<any[]>([]);

  useEffect(() => {
    if (!loading && !user) {
      router.push('/auth/login');
    } else if (user) {
      fetchRecentScans();
    }
  }, [user, loading, router]);

  const fetchRecentScans = async () => {
    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/api/ar-scanner/user-scans?user_id=${user?.id || 'demo_user'}`);
      const data = await response.json();
      if (data.success) {
        setRecentScans(data.scans);
      }
    } catch (error) {
      console.error('Failed to fetch scans:', error);
    }
  };

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
    }
  };

  const scanDocument = async () => {
    if (!selectedFile) return;

    setIsScanning(true);
    setScanResult(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);
      formData.append('user_id', user?.id || 'demo_user');
      formData.append('document_type', documentType);

      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/api/ar-scanner/scan-document`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      if (data.success) {
        setScanResult(data);
        fetchRecentScans(); // Refresh the list
      }
    } catch (error) {
      console.error('Scan failed:', error);
    } finally {
      setIsScanning(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <LoadingSpinner />
      </div>
    );
  }

  if (!user) {
    return null;
  }

  return (
    <DashboardLayout>
      <div className="max-w-6xl mx-auto space-y-6">
        {/* Header */}
        <div className="glass-strong rounded-2xl p-6 border border-dark-border-primary">
          <h1 className="text-3xl font-bold text-gradient">AR Medical Scanner</h1>
          <p className="text-dark-text-secondary mt-2">
            Scan and analyze medical documents with AI-powered OCR technology
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Scanner */}
          <div className="glass-strong rounded-2xl p-6 border border-dark-border-primary">
            <h2 className="text-xl font-semibold text-dark-text-primary mb-4">Scan Document</h2>
            
            {/* Document Type Selection */}
            <div className="mb-4">
              <label className="block text-sm font-medium text-dark-text-primary mb-2">
                Document Type
              </label>
              <select
                value={documentType}
                onChange={(e) => setDocumentType(e.target.value)}
                className="w-full bg-dark-bg-tertiary border border-dark-border-primary rounded-xl px-4 py-3 text-dark-text-primary focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all"
              >
                <option value="prescription">Prescription</option>
                <option value="lab_report">Lab Report</option>
                <option value="medical_record">Medical Record</option>
                <option value="insurance_card">Insurance Card</option>
              </select>
            </div>

            {/* File Upload */}
            <div className="mb-4">
              <label className="block text-sm font-medium text-dark-text-primary mb-2">
                Upload Document
              </label>
              <div className="border-2 border-dashed border-dark-border-primary rounded-xl p-8 text-center bg-dark-bg-secondary hover:border-primary-500 transition-all">
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleFileSelect}
                  className="hidden"
                  id="file-upload"
                />
                <label htmlFor="file-upload" className="cursor-pointer">
                  <svg className="mx-auto h-14 w-14 text-dark-text-secondary" stroke="currentColor" fill="none" viewBox="0 0 48 48">
                    <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                  <div className="mt-3">
                    <span className="text-primary-400 font-semibold">Click to upload</span>
                    <span className="text-dark-text-secondary"> or drag and drop</span>
                  </div>
                  <p className="text-xs text-dark-text-secondary mt-2">PNG, JPG, GIF up to 10MB</p>
                </label>
              </div>
              {selectedFile && (
                <p className="mt-3 text-sm text-dark-text-primary bg-dark-bg-tertiary px-3 py-2 rounded-lg border border-dark-border-primary">
                  Selected: {selectedFile.name}
                </p>
              )}
            </div>

            {/* Scan Button */}
            <button
              onClick={scanDocument}
              disabled={!selectedFile || isScanning}
              className="w-full bg-gradient-primary hover:opacity-90 disabled:opacity-50 text-white py-3 px-4 rounded-xl font-semibold shadow-lg glow-blue transition-all"
            >
              {isScanning ? (
                <div className="flex items-center justify-center">
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                  Scanning...
                </div>
              ) : (
                'Scan Document'
              )}
            </button>
          </div>

          {/* Results */}
          <div className="glass-strong rounded-2xl p-6 border border-dark-border-primary">
            <h2 className="text-xl font-semibold text-dark-text-primary mb-4">Scan Results</h2>
            
            {scanResult ? (
              <div className="space-y-4">
                <div>
                  <h3 className="font-semibold text-dark-text-primary mb-2">Extracted Text</h3>
                  <div className="bg-dark-bg-tertiary p-4 rounded-xl text-sm text-dark-text-primary border border-dark-border-primary">
                    {scanResult.extracted_text}
                  </div>
                </div>
                
                <div>
                  <h3 className="font-semibold text-dark-text-primary mb-3">AI Analysis</h3>
                  <div className="space-y-3">
                    <div className="flex items-center justify-between bg-dark-bg-tertiary p-3 rounded-lg border border-dark-border-primary">
                      <span className="text-sm text-dark-text-secondary">Confidence:</span>
                      <span className="text-sm font-semibold text-primary-400">{(scanResult.analysis.confidence_score * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex items-center justify-between bg-dark-bg-tertiary p-3 rounded-lg border border-dark-border-primary">
                      <span className="text-sm text-dark-text-secondary">Document Type:</span>
                      <span className="text-sm font-semibold text-dark-text-primary capitalize">{scanResult.analysis.document_type}</span>
                    </div>
                    
                    {scanResult.analysis.extracted_entities.medications.length > 0 && (
                      <div className="bg-dark-bg-tertiary p-4 rounded-xl border border-dark-border-primary">
                        <p className="text-sm font-semibold text-dark-text-primary mb-2">Medications:</p>
                        <ul className="text-sm text-dark-text-secondary ml-4 space-y-1">
                          {scanResult.analysis.extracted_entities.medications.map((med: string, index: number) => (
                            <li key={index}>• {med}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                    
                    {scanResult.analysis.warnings.length > 0 && (
                      <div className="bg-yellow-900 bg-opacity-30 border border-yellow-500 p-4 rounded-xl">
                        <p className="text-sm font-semibold text-yellow-400 mb-2">Warnings:</p>
                        <ul className="text-sm text-yellow-300 ml-4 space-y-1">
                          {scanResult.analysis.warnings.map((warning: string, index: number) => (
                            <li key={index}>• {warning}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center text-dark-text-secondary py-12">
                <svg className="mx-auto h-16 w-16 text-dark-text-tertiary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                <p className="mt-4 text-dark-text-secondary">Upload and scan a document to see results</p>
              </div>
            )}
          </div>
        </div>

        {/* Recent Scans */}
        <div className="glass-strong rounded-2xl p-6 border border-dark-border-primary">
          <h2 className="text-xl font-semibold text-dark-text-primary mb-4">Recent Scans</h2>
          
          {recentScans.length > 0 ? (
            <div className="space-y-3">
              {recentScans.map((scan, index) => (
                <div key={index} className="border border-dark-border-primary rounded-xl p-4 bg-dark-bg-tertiary hover:bg-dark-bg-hover transition-all">
                  <div className="flex justify-between items-start">
                    <div>
                      <p className="font-semibold text-dark-text-primary capitalize">
                        {scan.document_type.replace('_', ' ')}
                      </p>
                      <p className="text-sm text-dark-text-secondary mt-1">
                        {scan.extracted_text.substring(0, 100)}...
                      </p>
                      <p className="text-xs text-dark-text-tertiary mt-2">
                        {new Date(scan.created_at).toLocaleDateString()}
                      </p>
                    </div>
                    <span className="bg-gradient-to-r from-green-600 to-emerald-600 text-white text-xs px-3 py-1 rounded-full font-semibold shadow-lg">
                      {(scan.analysis.confidence_score * 100).toFixed(0)}% confidence
                    </span>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-dark-text-secondary text-center py-8">No scans yet</p>
          )}
        </div>
      </div>
    </DashboardLayout>
  );
};

export default ARScannerPage;