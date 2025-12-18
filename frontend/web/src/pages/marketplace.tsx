import React, { useState, useEffect } from 'react';
import { useRouter } from 'next/router';
import { useAuth } from '../contexts/AuthContext';
import DashboardLayout from '../components/layout/DashboardLayout';
import LoadingSpinner from '../components/ui/LoadingSpinner';

const MarketplacePage: React.FC = () => {
  const { user, loading } = useAuth();
  const router = useRouter();
  const [doctors, setDoctors] = useState<any[]>([]);
  const [filteredDoctors, setFilteredDoctors] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [filters, setFilters] = useState({
    specialty: '',
    location: '',
    availability: '',
    maxPrice: ''
  });
  const [selectedDoctor, setSelectedDoctor] = useState<any>(null);
  const [showBookingModal, setShowBookingModal] = useState(false);
  const [bookingData, setBookingData] = useState({
    appointment_date: '',
    appointment_time: '',
    symptoms: '',
    notes: '',
    consultation_type: 'in_person'
  });

  useEffect(() => {
    if (!loading && !user) {
      router.push('/auth/login');
    } else if (user) {
      fetchDoctors();
    }
  }, [user, loading, router]);

  const fetchDoctors = async () => {
    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const queryParams = new URLSearchParams();
      
      if (filters.specialty) queryParams.append('specialty', filters.specialty);
      if (filters.location) queryParams.append('location', filters.location);
      if (filters.availability) queryParams.append('availability', filters.availability);
      if (filters.maxPrice) queryParams.append('max_price', filters.maxPrice);

      const response = await fetch(`${apiUrl}/api/marketplace/doctors?${queryParams}`);
      const data = await response.json();
      
      if (data.success) {
        setDoctors(data.doctors);
        setFilteredDoctors(data.doctors);
      }
    } catch (error) {
      console.error('Failed to fetch doctors:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFilterChange = (key: string, value: string) => {
    const newFilters = { ...filters, [key]: value };
    setFilters(newFilters);
    
    // Apply filters
    let filtered = doctors;
    
    if (newFilters.specialty) {
      filtered = filtered.filter(doc => 
        doc.specialty.toLowerCase().includes(newFilters.specialty.toLowerCase())
      );
    }
    
    if (newFilters.location) {
      filtered = filtered.filter(doc => 
        doc.location.toLowerCase().includes(newFilters.location.toLowerCase())
      );
    }
    
    if (newFilters.availability === 'today') {
      filtered = filtered.filter(doc => 
        doc.availability.toLowerCase().includes('today')
      );
    }
    
    if (newFilters.maxPrice) {
      filtered = filtered.filter(doc => 
        doc.price <= parseInt(newFilters.maxPrice)
      );
    }
    
    setFilteredDoctors(filtered);
  };

  const openBookingModal = (doctor: any) => {
    setSelectedDoctor(doctor);
    setShowBookingModal(true);
  };

  const bookAppointment = async () => {
    if (!selectedDoctor) return;

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/api/marketplace/book-appointment`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          patient_id: user?.id || 'demo_user',
          doctor_id: selectedDoctor.id,
          appointment_date: bookingData.appointment_date,
          appointment_time: bookingData.appointment_time,
          symptoms: bookingData.symptoms.split(',').map(s => s.trim()),
          notes: bookingData.notes,
          price: selectedDoctor.price,
          consultation_type: bookingData.consultation_type
        }),
      });

      const data = await response.json();
      if (data.success) {
        alert('Appointment booked successfully!');
        setShowBookingModal(false);
        setBookingData({
          appointment_date: '',
          appointment_time: '',
          symptoms: '',
          notes: '',
          consultation_type: 'in_person'
        });
      }
    } catch (error) {
      console.error('Failed to book appointment:', error);
    }
  };

  if (loading || isLoading) {
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
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="glass-strong rounded-2xl p-6 border border-dark-border-primary">
          <h1 className="text-3xl font-bold text-gradient">Doctor Marketplace</h1>
          <p className="text-dark-text-secondary mt-2">
            Find and book appointments with healthcare specialists
          </p>
        </div>

        {/* Filters */}
        <div className="glass-strong rounded-2xl p-6 border border-dark-border-primary">
          <h2 className="text-xl font-semibold text-dark-text-primary mb-4">Find Your Doctor</h2>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div>
              <label className="block text-sm font-medium text-dark-text-primary mb-2">
                Specialty
              </label>
              <select
                value={filters.specialty}
                onChange={(e) => handleFilterChange('specialty', e.target.value)}
                className="w-full bg-dark-bg-tertiary border border-dark-border-primary rounded-xl px-4 py-3 text-dark-text-primary focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all"
              >
                <option value="">All Specialties</option>
                <option value="cardiology">Cardiology</option>
                <option value="dermatology">Dermatology</option>
                <option value="pediatrics">Pediatrics</option>
                <option value="orthopedics">Orthopedics</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-dark-text-primary mb-2">
                Location
              </label>
              <input
                type="text"
                value={filters.location}
                onChange={(e) => handleFilterChange('location', e.target.value)}
                placeholder="City, State"
                className="w-full bg-dark-bg-tertiary border border-dark-border-primary rounded-xl px-4 py-3 text-dark-text-primary placeholder-dark-text-secondary focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-dark-text-primary mb-2">
                Availability
              </label>
              <select
                value={filters.availability}
                onChange={(e) => handleFilterChange('availability', e.target.value)}
                className="w-full bg-dark-bg-tertiary border border-dark-border-primary rounded-xl px-4 py-3 text-dark-text-primary focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all"
              >
                <option value="">Any Time</option>
                <option value="today">Available Today</option>
                <option value="this_week">This Week</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-dark-text-primary mb-2">
                Max Price ($)
              </label>
              <input
                type="number"
                value={filters.maxPrice}
                onChange={(e) => handleFilterChange('maxPrice', e.target.value)}
                placeholder="200"
                className="w-full bg-dark-bg-tertiary border border-dark-border-primary rounded-xl px-4 py-3 text-dark-text-primary placeholder-dark-text-secondary focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all"
              />
            </div>
          </div>
        </div>

        {/* Doctors Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filteredDoctors.map((doctor) => (
            <div key={doctor.id} className="glass-strong rounded-2xl overflow-hidden border border-dark-border-primary hover:border-primary-500 transition-all">
              <div className="p-6">
                <div className="flex items-center space-x-4 mb-4">
                  <img
                    src={doctor.image}
                    alt={doctor.name}
                    className="w-16 h-16 rounded-full border-2 border-primary-500"
                  />
                  <div>
                    <h3 className="text-lg font-semibold text-dark-text-primary">{doctor.name}</h3>
                    <p className="text-sm text-dark-text-secondary">{doctor.specialty}</p>
                    <div className="flex items-center mt-1">
                      <div className="flex text-yellow-400">
                        {[...Array(5)].map((_, i) => (
                          <svg key={i} className="w-4 h-4 fill-current" viewBox="0 0 20 20">
                            <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                          </svg>
                        ))}
                      </div>
                      <span className="text-sm text-dark-text-secondary ml-1">
                        {doctor.rating} ({doctor.reviews} reviews)
                      </span>
                    </div>
                  </div>
                </div>

                <div className="space-y-2 mb-4">
                  <div className="flex justify-between text-sm bg-dark-bg-tertiary p-2 rounded-lg border border-dark-border-primary">
                    <span className="text-dark-text-secondary">Experience:</span>
                    <span className="font-semibold text-dark-text-primary">{doctor.experience}</span>
                  </div>
                  <div className="flex justify-between text-sm bg-dark-bg-tertiary p-2 rounded-lg border border-dark-border-primary">
                    <span className="text-dark-text-secondary">Location:</span>
                    <span className="font-semibold text-dark-text-primary">{doctor.location}</span>
                  </div>
                  <div className="flex justify-between text-sm bg-dark-bg-tertiary p-2 rounded-lg border border-dark-border-primary">
                    <span className="text-dark-text-secondary">Next Available:</span>
                    <span className="font-semibold text-green-400">{doctor.next_available}</span>
                  </div>
                  <div className="flex justify-between text-sm bg-dark-bg-tertiary p-2 rounded-lg border border-dark-border-primary">
                    <span className="text-dark-text-secondary">Consultation Fee:</span>
                    <span className="font-semibold text-primary-400">${doctor.price}</span>
                  </div>
                </div>

                <p className="text-sm text-dark-text-secondary mb-4">{doctor.bio}</p>

                <div className="flex space-x-2">
                  <button
                    onClick={() => openBookingModal(doctor)}
                    className="flex-1 bg-gradient-primary hover:opacity-90 text-white py-2 px-4 rounded-xl text-sm font-semibold shadow-lg glow-blue transition-all"
                  >
                    Book Appointment
                  </button>
                  <button className="px-4 py-2 border border-dark-border-primary rounded-xl text-sm font-semibold text-dark-text-primary hover:bg-dark-bg-hover transition-all bg-dark-bg-tertiary">
                    View Profile
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>

        {filteredDoctors.length === 0 && (
          <div className="text-center py-12 glass-strong rounded-2xl border border-dark-border-primary">
            <svg className="mx-auto h-16 w-16 text-dark-text-tertiary" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
            </svg>
            <h3 className="mt-4 text-lg font-semibold text-dark-text-primary">No doctors found</h3>
            <p className="mt-2 text-sm text-dark-text-secondary">Try adjusting your search filters.</p>
          </div>
        )}

        {/* Booking Modal */}
        {showBookingModal && selectedDoctor && (
          <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50">
            <div className="glass-strong rounded-2xl p-6 w-full max-w-md mx-4 border border-dark-border-primary">
              <div className="flex justify-between items-center mb-4">
                <h3 className="text-xl font-semibold text-dark-text-primary">
                  Book Appointment with {selectedDoctor.name}
                </h3>
                <button
                  onClick={() => setShowBookingModal(false)}
                  className="text-dark-text-secondary hover:text-dark-text-primary transition-colors"
                >
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>

              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-dark-text-primary mb-2">
                      Date
                    </label>
                    <input
                      type="date"
                      value={bookingData.appointment_date}
                      onChange={(e) => setBookingData({...bookingData, appointment_date: e.target.value})}
                      className="w-full bg-dark-bg-tertiary border border-dark-border-primary rounded-xl px-4 py-3 text-dark-text-primary focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-dark-text-primary mb-2">
                      Time
                    </label>
                    <input
                      type="time"
                      value={bookingData.appointment_time}
                      onChange={(e) => setBookingData({...bookingData, appointment_time: e.target.value})}
                      className="w-full bg-dark-bg-tertiary border border-dark-border-primary rounded-xl px-4 py-3 text-dark-text-primary focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all"
                    />
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-dark-text-primary mb-2">
                    Consultation Type
                  </label>
                  <select
                    value={bookingData.consultation_type}
                    onChange={(e) => setBookingData({...bookingData, consultation_type: e.target.value})}
                    className="w-full bg-dark-bg-tertiary border border-dark-border-primary rounded-xl px-4 py-3 text-dark-text-primary focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all"
                  >
                    <option value="in_person">In Person</option>
                    <option value="video">Video Call</option>
                    <option value="phone">Phone Call</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-dark-text-primary mb-2">
                    Symptoms (comma separated)
                  </label>
                  <input
                    type="text"
                    value={bookingData.symptoms}
                    onChange={(e) => setBookingData({...bookingData, symptoms: e.target.value})}
                    placeholder="headache, fever, fatigue"
                    className="w-full bg-dark-bg-tertiary border border-dark-border-primary rounded-xl px-4 py-3 text-dark-text-primary placeholder-dark-text-secondary focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-dark-text-primary mb-2">
                    Additional Notes
                  </label>
                  <textarea
                    value={bookingData.notes}
                    onChange={(e) => setBookingData({...bookingData, notes: e.target.value})}
                    rows={3}
                    className="w-full bg-dark-bg-tertiary border border-dark-border-primary rounded-xl px-4 py-3 text-dark-text-primary placeholder-dark-text-secondary focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent transition-all"
                    placeholder="Any additional information..."
                  />
                </div>

                <div className="bg-dark-bg-tertiary p-4 rounded-xl border border-dark-border-primary">
                  <div className="flex justify-between text-sm">
                    <span className="text-dark-text-secondary">Consultation Fee:</span>
                    <span className="font-semibold text-primary-400">${selectedDoctor.price}</span>
                  </div>
                </div>

                <div className="flex space-x-3">
                  <button
                    onClick={() => setShowBookingModal(false)}
                    className="flex-1 border border-dark-border-primary text-dark-text-primary py-3 px-4 rounded-xl font-semibold hover:bg-dark-bg-hover transition-all bg-dark-bg-tertiary"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={bookAppointment}
                    className="flex-1 bg-gradient-primary hover:opacity-90 text-white py-3 px-4 rounded-xl font-semibold shadow-lg glow-blue transition-all"
                  >
                    Book Appointment
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </DashboardLayout>
  );
};

export default MarketplacePage;