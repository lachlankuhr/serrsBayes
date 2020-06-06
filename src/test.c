#define LED_D1 (1 << 1)
#define LED_D2 (1 << 0)
#define LED_D3 (1 << 4)
#define LED_D4 (1 << 0)

int main(void)
{
    volatile uint32_t ui32Loop;

    // Enable the GPIO port that is used for the on-board LED.
    SYSCTL_RCGCGPIO_R = SYSCTL_RCGCGPIO_R12;

    // Short delay to allow peripheral to enable
    for(ui32Loop = 0; ui32Loop < 200000; ui32Loop++) { }

    // Enable GPIO pin for LED D1 & D2 (PN0 & PN1)
    GPIO_PORTN_DIR_R = LED_D2 | LED_D1;

    // Set the direction as output, and enable the GPIO pin for digital function.
    GPIO_PORTN_DEN_R = LED_D2 | LED_D1;
    GPIO_PORTN_DATA_R = LED_D1;

    // ** For extra task in Task D **** //

    SYSCTL_RCGCGPIO_R |= SYSCTL_RCGCGPIO_R5;

    // For extra task in Task D
    // Enable GPIO pin for LED D3 & D4 (PF0 & PF4)
    GPIO_PORTF_AHB_DIR_R = LED_D3 | LED_D4;

    // Set the direction as output, and enable the GPIO pin for digital function.
    GPIO_PORTF_AHB_DEN_R = LED_D3 | LED_D4;
    GPIO_PORTF_AHB_DATA_R = LED_D4;

    //**************************************//

    // Loop forever.
    while(1)
    {

        //Method using XOR bitwise operation (toggle bits)
        GPIO_PORTN_DATA_R ^= (LED_D1 | LED_D2);

        //for extra task in Task D
        GPIO_PORTF_AHB_DATA_R ^= (LED_D3 | LED_D4);

        // Delay for a bit.
        for(ui32Loop = 0; ui32Loop < 200000; ui32Loop++) { }


        // ** Alternative method *** //
        // Turn on LED D2.
        // GPIO_PORTN_DATA_R |= LED_D2;
        // Turn off the LED D1.
        // GPIO_PORTN_DATA_R &= ~(LED_D1);

        // Delay for a bit.
        // for(ui32Loop = 0; ui32Loop < 200000; ui32Loop++) { }

        // Turn on LED D1.
        // GPIO_PORTN_DATA_R |= LED_D1;
        // Turn off the LED D2.
        // GPIO_PORTN_DATA_R &= ~(LED_D2);

        // Delay for a bit.
        // for(ui32Loop = 0; ui32Loop < 200000; ui32Loop++) { }
    }
}