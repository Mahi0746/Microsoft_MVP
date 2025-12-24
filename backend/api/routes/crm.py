from typing import List, Optional
from fastapi import APIRouter, Request, HTTPException, status, Query, Depends
from pydantic import BaseModel
import structlog

from api.middleware.auth import get_current_user, require_role
from services.db_service import DatabaseService

logger = structlog.get_logger(__name__)
router = APIRouter()


class UserUpdateRequest(BaseModel):
    first_name: Optional[str]
    last_name: Optional[str]
    email: Optional[str]
    role: Optional[str]
    is_active: Optional[bool]


class AppointmentCreateRequest(BaseModel):
    patient_id: str
    doctor_id: Optional[str]
    symptoms_summary: Optional[str]
    urgency_level: Optional[str] = 'medium'
    consultation_type: Optional[str] = 'video'


class AppointmentUpdateRequest(BaseModel):
    status: Optional[str]
    doctor_id: Optional[str]
    scheduled_at: Optional[str]
    final_fee: Optional[float]


@router.get("/users", response_model=List[dict])
@require_role(['admin', 'doctor'])
async def list_users(request: Request, limit: int = Query(50, le=200), offset: int = 0, current_user: dict = Depends(get_current_user)):
    try:
        users = await DatabaseService.mongodb_find_many('users', {}, limit=limit)
        # Strip sensitive fields
        for u in users:
            u.pop('password_hash', None)
        return users
    except Exception as e:
        logger.error('Failed to list users', error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail='Failed to list users')


@router.get('/users/{user_id}')
@require_role(['admin', 'doctor'])
async def get_user(request: Request, user_id: str, current_user: dict = Depends(get_current_user)):
    try:
        user = await DatabaseService.mongodb_find_one('users', {'user_id': user_id})
        if not user:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='User not found')
        user.pop('password_hash', None)
        return user
    except HTTPException:
        raise
    except Exception as e:
        logger.error('Failed to get user', error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail='Failed to get user')


@router.put('/users/{user_id}')
@require_role(['admin', 'doctor'])
async def update_user(request: Request, user_id: str, body: UserUpdateRequest, current_user: dict = Depends(get_current_user)):
    try:
        update_data = {k: v for k, v in body.dict(exclude_unset=True).items()}
        if not update_data:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='No update fields')
        await DatabaseService.mongodb_update_one('users', {'user_id': user_id}, {'$set': update_data})
        user = await DatabaseService.mongodb_find_one('users', {'user_id': user_id})
        user.pop('password_hash', None)
        return user
    except HTTPException:
        raise
    except Exception as e:
        logger.error('Failed to update user', error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail='Failed to update user')


@router.delete('/users/{user_id}')
@require_role(['admin'])
async def delete_user(request: Request, user_id: str, current_user: dict = Depends(get_current_user)):
    try:
        deleted = await DatabaseService.mongodb_delete_one('users', {'user_id': user_id})
        if not deleted:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='User not found')
        return {'deleted': True}
    except HTTPException:
        raise
    except Exception as e:
        logger.error('Failed to delete user', error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail='Failed to delete user')


@router.get('/appointments', response_model=List[dict])
@require_role(['admin', 'doctor'])
async def list_appointments(request: Request, status_filter: Optional[str] = None, limit: int = Query(50, le=200), current_user: dict = Depends(get_current_user)):
    try:
        filt = {}
        if status_filter:
            filt['status'] = status_filter
        appts = await DatabaseService.mongodb_find_many('appointments', filt, limit=limit, sort=[('created_at', -1)])
        return appts
    except Exception as e:
        logger.error('Failed to list appointments', error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail='Failed to list appointments')


@router.get('/appointments/{appointment_id}')
@require_role(['admin', 'doctor'])
async def get_appointment(request: Request, appointment_id: str, current_user: dict = Depends(get_current_user)):
    try:
        appt = await DatabaseService.mongodb_find_one('appointments', {'appointment_id': appointment_id})
        if not appt:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Appointment not found')
        return appt
    except HTTPException:
        raise
    except Exception as e:
        logger.error('Failed to get appointment', error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail='Failed to get appointment')


@router.post('/appointments')
@require_role(['admin', 'doctor'])
async def create_appointment(request: Request, body: AppointmentCreateRequest, current_user: dict = Depends(get_current_user)):
    try:
        doc = body.dict()
        import uuid
        doc['appointment_id'] = str(uuid.uuid4())
        doc['status'] = 'pending'
        doc['created_at'] = __import__('datetime').datetime.utcnow()
        inserted_id = await DatabaseService.mongodb_insert_one('appointments', doc)
        created = await DatabaseService.mongodb_find_one('appointments', {'_id': inserted_id})
        return created
    except Exception as e:
        logger.error('Failed to create appointment', error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail='Failed to create appointment')


@router.put('/appointments/{appointment_id}')
@require_role(['admin', 'doctor'])
async def update_appointment(request: Request, appointment_id: str, body: AppointmentUpdateRequest, current_user: dict = Depends(get_current_user)):
    try:
        update_data = {k: v for k, v in body.dict(exclude_unset=True).items()}
        if not update_data:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='No update fields')
        await DatabaseService.mongodb_update_one('appointments', {'appointment_id': appointment_id}, {'$set': update_data})
        appt = await DatabaseService.mongodb_find_one('appointments', {'appointment_id': appointment_id})
        return appt
    except HTTPException:
        raise
    except Exception as e:
        logger.error('Failed to update appointment', error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail='Failed to update appointment')


@router.delete('/appointments/{appointment_id}')
@require_role(['admin'])
async def delete_appointment(request: Request, appointment_id: str, current_user: dict = Depends(get_current_user)):
    try:
        deleted = await DatabaseService.mongodb_delete_one('appointments', {'appointment_id': appointment_id})
        if not deleted:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail='Appointment not found')
        return {'deleted': True}
    except HTTPException:
        raise
    except Exception as e:
        logger.error('Failed to delete appointment', error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail='Failed to delete appointment')
